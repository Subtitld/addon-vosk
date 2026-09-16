"""Vosk ASR add-on — JSON-line protocol entry-point.

Wire format
-----------
Stdin and stdout carry one JSON object per line. Stderr is reserved for
free-form logging — the host prefixes each line with `[addon:org.subtitld.vosk]`
and surfaces it verbatim in its own stderr stream.

Lifecycle
---------
1. On startup, emit a `hello` frame advertising protocol version, addon id,
   version, and the `asr.transcribe` capability with the supported language
   tags. Block on stdin for the host's `ready` ack (host sends it within
   `manifest.startup_timeout_sec`).
2. Loop on stdin. Each `asr.transcribe` request triggers a worker that:
     a. Resolves the model for the requested language (downloading from
        alphacephei.com on first use unless the user set `auto_download=false`).
     b. Re-encodes the input to 16 kHz mono signed-16 WAV via ffmpeg (host
        passes the env var `SUBTITLD_FFMPEG_EXECUTABLE`; we fall back to
        `shutil.which('ffmpeg')`).
     c. Streams the WAV through `KaldiRecognizer.AcceptWaveform`, emitting one
        `partial` frame per finalized segment ({start, end, text, speaker}).
     d. Emits a terminal `result` frame with the full segments list.
3. `cancel` frames flip a worker-local flag — Kaldi is a synchronous C loop,
   so the cancel takes effect at the next AcceptWaveform call (≤ a few seconds).
4. `shutdown` causes a graceful exit.

Why threads instead of async? The Vosk Python bindings call into a C++
recognizer that holds the GIL across each AcceptWaveform — async wouldn't
help. A simple worker thread per request is fine; the host serialises
ASR requests to one provider at a time (see `AddonASRProvider.cancel`).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import Any

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel  # type: ignore
except ImportError as exc:  # pragma: no cover — surfaced via hello_error at runtime
    Model = None  # type: ignore[assignment]
    KaldiRecognizer = None  # type: ignore[assignment]
    SetLogLevel = None  # type: ignore[assignment]
    _VOSK_IMPORT_ERROR = exc
else:
    _VOSK_IMPORT_ERROR = None

ADDON_ID = 'vosk'
ADDON_VERSION = '1.0.1'
PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Error codes (mirrors `subtitld.modules.addons.protocol.ErrorCode`).
# Kept in-line to avoid taking a runtime dep on the host package — the addon
# binary must be self-contained.
# ---------------------------------------------------------------------------
ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK_UNAVAILABLE = 'network_unavailable'

log = logging.getLogger('vosk_addon')


# ---------------------------------------------------------------------------
# Static catalog of downloadable models. Mirrors `manifest.json:models[]` but
# lives in code too so the addon doesn't need to read its own manifest at
# runtime (the host is the source of truth for the manifest, but this addon
# is the source of truth for which model id maps to which URL).
# ---------------------------------------------------------------------------
_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    'vosk-model-small-en-us-0.15': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'language': 'en-us',
    },
    'vosk-model-en-us-0.22-lgraph': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'language': 'en-us',
    },
    'vosk-model-en-us-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'language': 'en-us',
    },
    'vosk-model-small-pt-0.3': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip',
        'language': 'pt-br',
    },
    'vosk-model-pt-fb-v0.1.1-20220516_2113': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip',
        'language': 'pt-br',
    },
    'vosk-model-small-es-0.42': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip',
        'language': 'es',
    },
    'vosk-model-es-0.42': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip',
        'language': 'es',
    },
    'vosk-model-small-fr-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip',
        'language': 'fr',
    },
    'vosk-model-fr-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip',
        'language': 'fr',
    },
    'vosk-model-small-de-0.15': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip',
        'language': 'de',
    },
    'vosk-model-de-0.21': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip',
        'language': 'de',
    },
    'vosk-model-small-it-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip',
        'language': 'it',
    },
    'vosk-model-it-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip',
        'language': 'it',
    },
    'vosk-model-small-ru-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip',
        'language': 'ru',
    },
    'vosk-model-ru-0.42': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip',
        'language': 'ru',
    },
    'vosk-model-small-cn-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip',
        'language': 'zh-cn',
    },
    'vosk-model-cn-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip',
        'language': 'zh-cn',
    },
    'vosk-model-small-ja-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip',
        'language': 'ja',
    },
    'vosk-model-ja-0.22': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip',
        'language': 'ja',
    },
}

# Per-language default. Lowest-bandwidth model that produces usable output;
# advanced users override via the addon's config UI.
_DEFAULT_MODEL_BY_LANGUAGE: dict[str, str] = {
    'en': 'vosk-model-small-en-us-0.15',
    'en-us': 'vosk-model-small-en-us-0.15',
    'pt': 'vosk-model-small-pt-0.3',
    'pt-br': 'vosk-model-small-pt-0.3',
    'es': 'vosk-model-small-es-0.42',
    'es-es': 'vosk-model-small-es-0.42',
    'fr': 'vosk-model-small-fr-0.22',
    'fr-fr': 'vosk-model-small-fr-0.22',
    'de': 'vosk-model-small-de-0.15',
    'de-de': 'vosk-model-small-de-0.15',
    'it': 'vosk-model-small-it-0.22',
    'it-it': 'vosk-model-small-it-0.22',
    'ru': 'vosk-model-small-ru-0.22',
    'zh': 'vosk-model-small-cn-0.22',
    'zh-cn': 'vosk-model-small-cn-0.22',
    'ja': 'vosk-model-small-ja-0.22',
}

_SUPPORTED_LANGUAGES = sorted(set(_DEFAULT_MODEL_BY_LANGUAGE.keys()))


# ---------------------------------------------------------------------------
# Frame encode / decode. Inlined (no import from host package) so the frozen
# binary stays standalone.
# ---------------------------------------------------------------------------
_stdout_lock = threading.Lock()


def _emit(frame: dict) -> None:
    """Write one frame to stdout. Thread-safe — multiple worker threads may
    call this concurrently."""
    line = json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n'
    data = line.encode('utf-8')
    with _stdout_lock:
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            # Host went away. Nothing to do; main loop will exit on the next
            # readline() returning b''.
            pass


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress', 'value': max(0.0, min(1.0, value)), 'message': message})


def _emit_partial(req_id: str, segment: dict) -> None:
    _emit({'id': req_id, 'type': 'partial', 'data': {'segment': segment}})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({
        'id': req_id,
        'type': 'error',
        'code': code,
        'message': message,
        'retryable': retryable,
    })


# ---------------------------------------------------------------------------
# Filesystem layout. Kept under the addon's user data dir so an uninstall
# can drop it in one go. Mirrors what `platformdirs.user_data_dir('subtitld')`
# would resolve to, but reimplemented to keep the addon dep-free.
# ---------------------------------------------------------------------------
def _user_data_dir() -> Path:
    """Return `<user-data>/subtitld/addons/org.subtitld.vosk/`.

    Honors `SUBTITLD_VOSK_DATA_DIR` for tests and corner cases (e.g. a sandbox
    where the canonical path isn't writable).
    """
    override = os.environ.get('SUBTITLD_VOSK_DATA_DIR')
    if override:
        return Path(override)

    if sys.platform.startswith('linux'):
        base = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    elif sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support')
    elif sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    else:
        base = os.path.join(os.path.expanduser('~'), '.local', 'share')

    return Path(base) / 'subtitld' / 'addons' / ADDON_ID


def _models_dir() -> Path:
    p = _user_data_dir() / 'models'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _temp_dir() -> Path:
    p = _user_data_dir() / 'tmp'
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# ffmpeg discovery. The host sets `SUBTITLD_FFMPEG_EXECUTABLE` for built-ins,
# but we don't currently propagate it across the AddonProcess boundary — fall
# back to system PATH and (last resort) error.
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    return (
        os.environ.get('SUBTITLD_FFMPEG_EXECUTABLE')
        or os.environ.get('FFMPEG_EXECUTABLE')
        or shutil.which('ffmpeg')
    )


def _convert_to_pcm16k_mono(src: str, dst: str) -> None:
    """Re-encode any input to 16 kHz mono signed-16 WAV. Vosk requires this
    exact format; sending anything else either errors or produces garbage."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found on PATH and SUBTITLD_FFMPEG_EXECUTABLE not set')

    # Hide the console window when this addon is built for Windows and run
    # under a GUI host. Without this each ffmpeg call flashes a cmd.exe.
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE

    # `-nostdin` plus stdin=DEVNULL: ffmpeg polls stdin for keystrokes, and
    # ours carries host frames — a `cancel` sent mid-conversion would vanish.
    proc = subprocess.run(
        [
            ffmpeg, '-nostdin', '-y',
            '-i', src,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            dst,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        startupinfo=startupinfo,
    )
    if proc.returncode != 0:
        # ffmpeg's stderr is the only useful diagnostic; trim to last few
        # lines so we don't flood the protocol channel.
        tail = proc.stderr.decode('utf-8', errors='replace').strip().splitlines()[-5:]
        raise RuntimeError(f'ffmpeg failed (rc={proc.returncode}): ' + ' | '.join(tail))


# ---------------------------------------------------------------------------
# Model resolution + download. Models live unzipped at
# `<models_dir>/<model_id>/`. The Vosk `Model(path)` API takes the directory.
# ---------------------------------------------------------------------------
def _resolve_model_id(language: str, options: dict) -> str | None:
    """Pick a model id given the language and per-request options.

    Resolution order:
        1. `options.model` if present and known.
        2. `_DEFAULT_MODEL_BY_LANGUAGE[language]`.
        3. `_DEFAULT_MODEL_BY_LANGUAGE[language[:2]]` (strip region tag).
    """
    explicit = options.get('model') if isinstance(options, dict) else None
    if explicit and explicit in _MODEL_CATALOG:
        return explicit
    lang = (language or '').strip().lower()
    if lang in _DEFAULT_MODEL_BY_LANGUAGE:
        return _DEFAULT_MODEL_BY_LANGUAGE[lang]
    if '-' in lang:
        return _DEFAULT_MODEL_BY_LANGUAGE.get(lang.split('-', 1)[0])
    return None


def _model_path(model_id: str) -> Path:
    return _models_dir() / model_id


def _ensure_model(model_id: str, *, auto_download: bool, req_id: str, cancel_flag: threading.Event) -> Path:
    """Return the path to an installed model, downloading it if necessary.

    Raises `RuntimeError` with a stable error code prefix that the caller
    decodes back into the protocol's `error.code`. Encoding it that way keeps
    the recursion shallow — one helper that knows about both filesystem and
    network errors.
    """
    target = _model_path(model_id)
    if target.is_dir() and any(target.iterdir()):
        return target

    if not auto_download:
        raise RuntimeError(f'{ERR_MODEL_MISSING}: model {model_id!r} not installed and auto_download=false')

    catalog_entry = _MODEL_CATALOG.get(model_id)
    if not catalog_entry:
        raise RuntimeError(f'{ERR_MODEL_MISSING}: unknown model id {model_id!r}')

    url = catalog_entry['url']
    log.info('downloading %s from %s', model_id, url)
    _emit_progress(req_id, 0.0, f'Downloading {model_id} …')

    tmp_dir = _temp_dir()
    zip_tmp = tmp_dir / f'{model_id}.zip.part'
    extract_tmp = tmp_dir / f'{model_id}.extracting'

    if extract_tmp.exists():
        shutil.rmtree(extract_tmp, ignore_errors=True)

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            downloaded = 0
            chunk = 1024 * 1024  # 1 MiB
            with open(zip_tmp, 'wb') as fh:
                while True:
                    if cancel_flag.is_set():
                        raise RuntimeError(f'{ERR_CANCELLED}: download cancelled')
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
                    if total:
                        # Reserve 0..0.5 of the 0..1 progress range for the
                        # download; the recogniser run gets the remaining 0.5.
                        _emit_progress(req_id, 0.5 * (downloaded / total),
                                       f'Downloading {model_id} ({downloaded // (1024 * 1024)}/{total // (1024 * 1024)} MiB)')
    except urllib.error.URLError as exc:
        raise RuntimeError(f'{ERR_NETWORK_UNAVAILABLE}: {exc.reason}') from exc
    except OSError as exc:
        # ENOSPC etc.
        if getattr(exc, 'errno', None) == 28:  # ENOSPC
            raise RuntimeError(f'{ERR_DISK_FULL}: {exc}') from exc
        raise RuntimeError(f'{ERR_INTERNAL}: download failed: {exc}') from exc

    _emit_progress(req_id, 0.5, f'Extracting {model_id} …')

    try:
        extract_tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_tmp) as zf:
            zf.extractall(extract_tmp)
    except (zipfile.BadZipFile, OSError) as exc:
        shutil.rmtree(extract_tmp, ignore_errors=True)
        raise RuntimeError(f'{ERR_INTERNAL}: extraction failed: {exc}') from exc
    finally:
        try:
            zip_tmp.unlink()
        except OSError:
            pass

    # Vosk model zips contain a single top-level folder (`vosk-model-small-en-us-0.15/...`).
    # Promote it to `<models>/<model_id>/` for a stable install path.
    children = [p for p in extract_tmp.iterdir() if p.is_dir()]
    if len(children) == 1 and children[0].name == model_id:
        # Atomic rename — model directory becomes visible only when complete.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        children[0].rename(target)
        shutil.rmtree(extract_tmp, ignore_errors=True)
    else:
        # Some zips don't follow the convention; fall back to renaming the
        # whole staging dir.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        extract_tmp.rename(target)

    return target


# ---------------------------------------------------------------------------
# Model cache. Keep at most one Model in memory at a time — they're large
# (50 MB ↗ 2 GB) and the host's idle-GC will tear us down anyway.
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {'id': None, 'instance': None}
_model_cache_lock = threading.Lock()


def _load_model(model_path: Path):
    if Model is None:
        raise RuntimeError(f'{ERR_INTERNAL}: vosk Python bindings missing: {_VOSK_IMPORT_ERROR!r}')
    with _model_cache_lock:
        cached_id = _model_cache.get('id')
        if cached_id == str(model_path) and _model_cache.get('instance') is not None:
            return _model_cache['instance']
        # Drop the previous instance before loading a new one. The C++ object
        # frees memory in its destructor; without this we'd briefly hold both.
        _model_cache['id'] = None
        _model_cache['instance'] = None
        instance = Model(str(model_path))
        _model_cache['id'] = str(model_path)
        _model_cache['instance'] = instance
        return instance


# ---------------------------------------------------------------------------
# Worker: one ASR request lifecycle.
# ---------------------------------------------------------------------------
class _WorkerState:
    """Per-request handle the dispatcher hands to the worker thread.

    Holds the cancel flag (shared with the dispatcher's `cancel` handler) and
    the request id. Plain dataclass would be enough but threading.Event is
    nicer to test."""

    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel_flag = threading.Event()
        self.thread: threading.Thread | None = None


_active_requests_lock = threading.Lock()
_active_requests: dict[str, _WorkerState] = {}


def _spawn_worker(req_id: str, target, *args) -> None:
    state = _WorkerState(req_id)
    with _active_requests_lock:
        _active_requests[req_id] = state

    def _run():
        try:
            target(state, *args)
        except Exception:
            log.exception('worker for request %s crashed', req_id)
            tb = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tb}')
        finally:
            with _active_requests_lock:
                _active_requests.pop(req_id, None)

    t = threading.Thread(target=_run, name=f'vosk-worker-{req_id}', daemon=True)
    state.thread = t
    t.start()


def _handle_transcribe(state: _WorkerState, params: dict) -> None:
    req_id = state.req_id

    audio_path = params.get('audio_path')
    language = (params.get('language') or '').strip().lower()
    options = params.get('options') or {}
    auto_download = options.get('auto_download', True)

    if not audio_path or not os.path.isfile(audio_path):
        _emit_error(req_id, ERR_BAD_PARAMS, f'audio_path missing or not a file: {audio_path!r}')
        return

    model_id = _resolve_model_id(language, options)
    if model_id is None:
        _emit_error(req_id, ERR_UNSUPPORTED_LANGUAGE,
                    f'no Vosk model registered for language {language!r}')
        return

    # Stage 1 — ensure model on disk (may download).
    try:
        model_dir = _ensure_model(
            model_id,
            auto_download=bool(auto_download),
            req_id=req_id,
            cancel_flag=state.cancel_flag,
        )
    except RuntimeError as exc:
        msg = str(exc)
        # Helper raises with `code: detail`. Split safely.
        code, _, detail = msg.partition(':')
        code = code.strip() or ERR_INTERNAL
        _emit_error(req_id, code, detail.strip() or msg, retryable=(code == ERR_NETWORK_UNAVAILABLE))
        return

    if state.cancel_flag.is_set():
        _emit_error(req_id, ERR_CANCELLED, 'cancelled before model load')
        return

    # Stage 2 — convert audio to 16 kHz mono PCM16 WAV.
    _emit_progress(req_id, 0.55, 'Preparing audio …')
    wav_path = _temp_dir() / f'transcribe_{req_id}.wav'
    try:
        _convert_to_pcm16k_mono(audio_path, str(wav_path))
    except Exception as exc:
        _emit_error(req_id, ERR_INTERNAL, f'audio conversion failed: {exc}')
        return

    if state.cancel_flag.is_set():
        wav_path.unlink(missing_ok=True)
        _emit_error(req_id, ERR_CANCELLED, 'cancelled before recognition')
        return

    # Stage 3 — load model + run KaldiRecognizer.
    try:
        model = _load_model(model_dir)
    except Exception as exc:
        _emit_error(req_id, ERR_INTERNAL, f'failed to load model: {exc}')
        return

    if state.cancel_flag.is_set():
        wav_path.unlink(missing_ok=True)
        _emit_error(req_id, ERR_CANCELLED, 'cancelled before recognition')
        return

    speaker_default = options.get('speaker') or 'A'
    segments: list[dict] = []
    audio_duration = 0.0

    try:
        with wave.open(str(wav_path), 'rb') as wf:
            framerate = wf.getframerate()
            total_frames = wf.getnframes()
            audio_duration = total_frames / float(framerate) if framerate else 0.0

            recognizer = KaldiRecognizer(model, framerate)
            recognizer.SetWords(True)

            chunk = 4000
            frames_read = 0
            _emit_progress(req_id, 0.6, f'Recognising ({int(audio_duration)}s) …')

            while True:
                if state.cancel_flag.is_set():
                    _emit_error(req_id, ERR_CANCELLED, 'cancelled mid-recognition')
                    return

                data = wf.readframes(chunk)
                if not data:
                    break
                frames_read += chunk

                if recognizer.AcceptWaveform(data):
                    segment = _parse_kaldi_result(recognizer.Result(), speaker_default)
                    if segment:
                        segments.append(segment)
                        _emit_partial(req_id, segment)

                if total_frames and audio_duration:
                    # 0.6 .. 0.99 reserved for the recognition pass.
                    ratio = min(1.0, frames_read / total_frames)
                    _emit_progress(req_id, 0.6 + 0.39 * ratio,
                                   f'Recognising ({int(ratio * audio_duration)}/{int(audio_duration)}s)')

            # Flush any remaining buffered audio into a final segment.
            final_segment = _parse_kaldi_result(recognizer.FinalResult(), speaker_default)
            if final_segment:
                segments.append(final_segment)
                _emit_partial(req_id, final_segment)
    except Exception as exc:
        _emit_error(req_id, ERR_INTERNAL, f'recognition failed: {exc}')
        return
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass

    _emit_progress(req_id, 1.0, f'Done — {len(segments)} segment(s)')
    _emit_result(req_id, {
        'segments': segments,
        'duration_sec': audio_duration,
        'model': model_id,
    })


# ---------------------------------------------------------------------------
# Kaldi result parsing. Vosk emits JSON with `text` (joined utterance) and
# `result` (per-word [{conf, start, end, word}, ...]). We collapse the word
# list into one segment with start/end taken from the first/last word and the
# joined text as the segment text.
# ---------------------------------------------------------------------------
_DECIMAL_COMMA_RE = re.compile(r'(\d),(\d)')


def _parse_kaldi_result(raw: str, speaker: str) -> dict | None:
    """Parse one `Result()`/`FinalResult()` JSON blob into a segment dict.

    Returns None when the result has no words (Kaldi emits empty-text frames
    for silence). Locale-quirk: Vosk's `pt-fb` model occasionally writes
    decimal commas where it should write decimal points (e.g. "0,12"); the
    legacy host code worked around this by replacing them before json.loads,
    so we preserve the workaround.
    """
    if not raw:
        return None
    fixed = _DECIMAL_COMMA_RE.sub(r'\1.\2', raw)
    try:
        parsed = json.loads(fixed)
    except json.JSONDecodeError:
        return None
    words = parsed.get('result') or []
    text = (parsed.get('text') or '').strip()
    if not words or not text:
        return None
    return {
        'start': float(words[0].get('start', 0.0)),
        'end': float(words[-1].get('end', 0.0)),
        'text': text,
        'speaker': speaker,
    }


# ---------------------------------------------------------------------------
# Dispatcher loop. Reads from stdin, dispatches to handlers.
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [
            {
                'task': 'asr.transcribe',
                'languages': _SUPPORTED_LANGUAGES,
                'streaming': True,
                'voice_clone': False,
            },
        ],
    })


def _send_hello_error(message: str, code: str = ERR_INTERNAL) -> None:
    _emit({'type': 'hello_error', 'code': code, 'message': message})


def _handle_cancel(target_id: str) -> None:
    with _active_requests_lock:
        state = _active_requests.get(target_id)
    if state is None:
        log.info('cancel: no active request %s', target_id)
        return
    log.info('cancel: signalling worker for %s', target_id)
    state.cancel_flag.set()


def _read_frame_blocking() -> dict | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None  # stdin closed → host went away
    try:
        text = line.decode('utf-8', errors='replace').strip().lstrip('\ufeff')
        if not text:
            return {}  # empty line — caller skips
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning('malformed frame %r: %s', line, exc)
        return {}


def _handshake() -> bool:
    """Perform the hello/ready handshake. Returns True if the host accepted us."""
    if _VOSK_IMPORT_ERROR is not None:
        _send_hello_error(
            f'vosk Python bindings missing: {_VOSK_IMPORT_ERROR}. '
            'Reinstall the add-on (the prebuilt zip should bundle vosk).'
        )
        return False

    if _find_ffmpeg() is None:
        # Not fatal at handshake — only fatal once we get a transcribe
        # request. We still want the host to register us so the user sees
        # the addon and gets a clear error in the UI rather than a silent
        # disappearance from the engine list.
        log.warning('ffmpeg not found at startup — transcription requests will fail until one is on PATH')

    _send_hello()

    # Wait for `ready`. Fail-fast on any other type — host shouldn't be sending
    # requests before acking the handshake.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        frame = _read_frame_blocking()
        if frame is None:
            return False
        if not frame:
            continue
        if frame.get('type') == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            return True
        log.warning('unexpected pre-ready frame: %r', frame)

    log.warning('host did not send ready within 30s')
    return False


def _main_loop() -> int:
    while True:
        frame = _read_frame_blocking()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            # Signal all workers; they'll exit at their next checkpoint.
            with _active_requests_lock:
                states = list(_active_requests.values())
            for state in states:
                state.cancel_flag.set()
            # Best-effort drain.
            for state in states:
                if state.thread is not None:
                    state.thread.join(timeout=4.0)
            return 0

        if ftype == 'cancel':
            target = frame.get('target')
            if isinstance(target, str):
                _handle_cancel(target)
            continue

        if ftype == 'asr.transcribe':
            req_id = frame.get('id')
            params = frame.get('params') or {}
            if not isinstance(req_id, str) or not req_id:
                # Without an id we can't even respond; just log and drop.
                log.warning('asr.transcribe without id: %r', frame)
                continue
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn_worker(req_id, _handle_transcribe, params)
            continue

        # Unknown task. Still answer with `error` so the host's request future
        # resolves rather than waiting forever.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def main() -> int:
    # Keep the C++ recogniser from spamming stderr — addon stderr already
    # gets prefixed and surfaced by the host, but Vosk's default level is
    # very chatty (Kaldi diagnostics on every model load).
    if SetLogLevel is not None:
        try:
            SetLogLevel(-1)
        except Exception:
            pass

    logging.basicConfig(
        level=os.environ.get('VOSK_ADDON_LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    log.info('vosk-addon %s starting on %s %s (python %s)',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())
    log.info('user-data dir: %s', _user_data_dir())

    if not _handshake():
        return 1

    try:
        return _main_loop()
    except KeyboardInterrupt:
        log.info('interrupted')
        return 130
    except Exception:
        log.exception('main loop crashed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
