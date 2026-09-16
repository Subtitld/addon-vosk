"""Smoke tests for `vosk_addon` that don't require a real Vosk model.

We exercise:

  * Frame encode/decode via `_emit` capturing stdout in a BytesIO.
  * `_resolve_model_id` for the language → model mapping.
  * `_parse_kaldi_result` for Vosk's locale-comma quirk and empty-result
    behaviour.
  * Error path of `_handle_transcribe` for missing audio + unsupported
    language (no model load needed).

Running the full handshake/dispatcher loop requires a subprocess driver;
that's out of scope for this smoke set — the host's integration tests
take care of it once the binary is built.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Allow `pytest` from the addon root without an editable install — the
# tests live next to `vosk_addon/`, so adding the parent dir to sys.path
# is enough.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vosk_addon.__main__ as addon  # noqa: E402


@pytest.fixture(autouse=True)
def _capture_frames(monkeypatch):
    """Replace `_emit` with a list-appending stub.

    Patching `_emit` (rather than `sys.stdout`) sidesteps pytest's stdout
    capture, which otherwise intercepts the file-descriptor-level writes
    we'd get if we let `_emit` reach the real `sys.stdout.buffer`. The
    helpers (`_emit_progress`, `_emit_partial`, `_emit_result`,
    `_emit_error`) all funnel through `_emit`, so a single hook is enough
    to observe every frame any code path produces.
    """
    captured: list[dict] = []
    monkeypatch.setattr(addon, '_emit', captured.append)
    yield captured


def _read_frames(captured: list[dict]) -> list[dict]:
    """No-op for legacy compatibility with the old BytesIO-based reader.

    The addon emits frames as plain dicts now (since we patch `_emit`
    upstream of the JSON serialisation), so the test just hands the list
    back. Kept as a function so existing call-sites read uniformly.
    """
    return list(captured)


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def test_emit_progress_emits_clamped_value(_capture_frames):
    addon._emit_progress('r1', 1.5, 'over the top')
    addon._emit_progress('r1', -0.2, 'under the floor')
    addon._emit_progress('r1', 0.42, 'just right')

    frames = _read_frames(_capture_frames)
    assert [f['value'] for f in frames] == [1.0, 0.0, 0.42]
    assert all(f['type'] == 'progress' for f in frames)
    assert all(f['id'] == 'r1' for f in frames)


def test_emit_partial_wraps_segment_under_data(_capture_frames):
    seg = {'start': 0.1, 'end': 0.5, 'text': 'hello', 'speaker': 'A'}
    addon._emit_partial('r1', seg)
    frames = _read_frames(_capture_frames)
    assert frames == [{'id': 'r1', 'type': 'partial', 'data': {'segment': seg}}]


def test_emit_error_includes_code_and_retryable(_capture_frames):
    addon._emit_error('r1', addon.ERR_NETWORK_UNAVAILABLE, 'flaky wifi', retryable=True)
    frames = _read_frames(_capture_frames)
    assert frames[0] == {
        'id': 'r1',
        'type': 'error',
        'code': 'network_unavailable',
        'message': 'flaky wifi',
        'retryable': True,
    }


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('language,expected', [
    ('en-us', 'vosk-model-small-en-us-0.15'),
    ('en', 'vosk-model-small-en-us-0.15'),
    ('pt-br', 'vosk-model-small-pt-0.3'),
    ('pt', 'vosk-model-small-pt-0.3'),
    ('PT-BR', 'vosk-model-small-pt-0.3'),         # case-insensitive
    ('zh-cn', 'vosk-model-small-cn-0.22'),
    ('fr-ca', 'vosk-model-small-fr-0.22'),         # region fallback to base
    ('xx', None),                                   # unsupported
])
def test_resolve_model_id_defaults(language, expected):
    assert addon._resolve_model_id(language, {}) == expected


def test_resolve_model_id_explicit_override():
    # Explicit `model` option wins over the per-language default.
    got = addon._resolve_model_id('en-us', {'model': 'vosk-model-en-us-0.22-lgraph'})
    assert got == 'vosk-model-en-us-0.22-lgraph'


def test_resolve_model_id_explicit_unknown_falls_back_to_default():
    got = addon._resolve_model_id('en-us', {'model': 'not-a-real-model'})
    assert got == 'vosk-model-small-en-us-0.15'


# ---------------------------------------------------------------------------
# Kaldi result parsing
# ---------------------------------------------------------------------------
def test_parse_kaldi_result_basic():
    raw = json.dumps({
        'text': 'hello world',
        'result': [
            {'word': 'hello', 'start': 0.10, 'end': 0.45, 'conf': 0.99},
            {'word': 'world', 'start': 0.50, 'end': 1.20, 'conf': 0.97},
        ],
    })
    seg = addon._parse_kaldi_result(raw, 'A')
    assert seg == {'start': 0.10, 'end': 1.20, 'text': 'hello world', 'speaker': 'A'}


def test_parse_kaldi_result_handles_decimal_comma():
    # The legacy host code worked around Vosk-pt-fb emitting commas in
    # decimal numbers — we keep the same regex fix here.
    raw = '{"text":"oi","result":[{"word":"oi","start":0,12,"end":0,33,"conf":0,99}]}'
    seg = addon._parse_kaldi_result(raw, 'A')
    assert seg is not None
    assert seg['start'] == pytest.approx(0.12)
    assert seg['end'] == pytest.approx(0.33)


def test_parse_kaldi_result_returns_none_on_silence():
    # Vosk emits empty text/result for pure-silence chunks — must skip.
    assert addon._parse_kaldi_result('{"text":"","result":[]}', 'A') is None
    assert addon._parse_kaldi_result('', 'A') is None
    assert addon._parse_kaldi_result('not json', 'A') is None


# ---------------------------------------------------------------------------
# Worker error paths (no model load required)
# ---------------------------------------------------------------------------
def _await_thread(state: addon._WorkerState):
    """Helper — wait for the worker to finish, with a hard ceiling so a
    deadlock fails the test instead of hanging the suite."""
    if state.thread is not None:
        state.thread.join(timeout=10.0)
        assert not state.thread.is_alive(), 'worker did not exit in time'


def test_transcribe_missing_audio_emits_bad_params(_capture_frames, tmp_path):
    nonexistent = tmp_path / 'no-such-file.wav'
    state = addon._WorkerState('r1')
    addon._handle_transcribe(state, {
        'audio_path': str(nonexistent),
        'language': 'en-us',
        'options': {},
    })
    frames = _read_frames(_capture_frames)
    assert frames[-1]['type'] == 'error'
    assert frames[-1]['code'] == 'bad_params'


def test_transcribe_unsupported_language_emits_unsupported_language(_capture_frames, tmp_path):
    # Provide a real (empty) file so we get past the audio_path check.
    fake = tmp_path / 'fake.wav'
    fake.write_bytes(b'\x00' * 16)
    state = addon._WorkerState('r1')
    addon._handle_transcribe(state, {
        'audio_path': str(fake),
        'language': 'xx',
        'options': {},
    })
    frames = _read_frames(_capture_frames)
    assert frames[-1]['type'] == 'error'
    assert frames[-1]['code'] == 'unsupported_language'


def test_transcribe_missing_model_with_auto_download_off_emits_model_missing(
    _capture_frames, tmp_path, monkeypatch
):
    fake = tmp_path / 'fake.wav'
    fake.write_bytes(b'\x00' * 16)
    # Sandbox the data dir so the test never touches the real one.
    monkeypatch.setenv('SUBTITLD_VOSK_DATA_DIR', str(tmp_path / 'addon-data'))
    state = addon._WorkerState('r1')
    addon._handle_transcribe(state, {
        'audio_path': str(fake),
        'language': 'en-us',
        'options': {'auto_download': False},
    })
    frames = _read_frames(_capture_frames)
    assert frames[-1]['type'] == 'error'
    assert frames[-1]['code'] == 'model_missing'


# ---------------------------------------------------------------------------
# Hello frame shape
# ---------------------------------------------------------------------------
def test_send_hello_advertises_supported_languages(_capture_frames):
    addon._send_hello()
    frames = _read_frames(_capture_frames)
    assert len(frames) == 1
    hello = frames[0]
    assert hello['type'] == 'hello'
    assert hello['protocol'] == 1
    assert hello['addon'] == 'vosk'
    assert any(c['task'] == 'asr.transcribe' for c in hello['capabilities'])
    asr_cap = [c for c in hello['capabilities'] if c['task'] == 'asr.transcribe'][0]
    # Sanity-check: we must advertise at least the languages from the default
    # mapping, otherwise the host would filter us out for those locales.
    for lang in ('en-us', 'pt-br', 'zh-cn', 'ja'):
        assert lang in asr_cap['languages']


# ---------------------------------------------------------------------------
# ffmpeg must not read the protocol channel
# ---------------------------------------------------------------------------
# Stand-in for ffmpeg. Once the test has queued host frames it takes every
# byte its stdin offers — the worst case of ffmpeg's keyboard polling, which
# eats queued frames whenever it shares the add-on's stdin.
_STDIN_EATER = '''
import json, os, select, sys, time
from pathlib import Path

work = Path(os.environ['STUB_FFMPEG_DIR'])
(work / 'started').touch()
deadline = time.monotonic() + 10
while not (work / 'go').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
consumed = 0
while select.select([0], [], [], 0.5)[0]:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    consumed += len(chunk)
(work / 'report.json').write_text(json.dumps({'argv': sys.argv[1:], 'consumed': consumed}))
Path(sys.argv[-1]).write_bytes(b'RIFF stub')
'''


@contextlib.contextmanager
def _host_pipe_as_stdin():
    """Put a fresh pipe on fd 0, where host frames arrive; yield its write end."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(0)
    os.dup2(read_fd, 0)
    os.close(read_fd)
    host = os.fdopen(write_fd, 'wb')
    try:
        yield host
    finally:
        host.close()
        os.dup2(saved, 0)
        os.close(saved)


def _wait_until_started(marker, job):
    deadline = time.monotonic() + 10
    while not marker.exists():
        if job.done():
            job.result()  # surfaces the helper's own error
            raise AssertionError('helper returned without running the stand-in')
        assert time.monotonic() < deadline, 'stand-in never started'
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason='the stand-in is a shebang script')
def test_ffmpeg_does_not_swallow_queued_host_frames(tmp_path, monkeypatch):
    """A `cancel` sent while audio converts must still reach the dispatcher.

    Driven in-process: with the real dispatcher thread also reading stdin,
    it races ffmpeg for each frame and a regression would only fail
    sometimes. Here the stand-in is the only other reader, so it fails
    every time.
    """
    stub = tmp_path / 'ffmpeg'
    stub.write_text(f'#!{sys.executable}\n{_STDIN_EATER}', encoding='utf-8')
    stub.chmod(0o755)
    monkeypatch.setenv('SUBTITLD_FFMPEG_EXECUTABLE', str(stub))
    monkeypatch.setenv('STUB_FFMPEG_DIR', str(tmp_path))
    src = tmp_path / 'in.mp3'
    src.write_bytes(b'ID3')

    sent = [
        {'type': 'cancel', 'target': 'r1'},
        {'type': 'asr.transcribe', 'id': 'r2', 'params': {'audio_path': str(src)}},
        {'type': 'cancel', 'target': 'r2'},
        {'type': 'shutdown'},
    ]
    received = []
    with _host_pipe_as_stdin() as host, ThreadPoolExecutor(1) as pool:
        job = pool.submit(addon._convert_to_pcm16k_mono, str(src), str(tmp_path / 'out.wav'))
        _wait_until_started(tmp_path / 'started', job)
        for frame in sent:
            host.write((json.dumps(frame) + '\n').encode())
        host.close()
        (tmp_path / 'go').touch()
        job.result(timeout=30)

        with open(0, encoding='utf-8', closefd=False) as stdin:
            monkeypatch.setattr(sys, 'stdin', stdin)
            while (frame := addon._read_frame_blocking()) is not None:
                received.append(frame)

    assert received == sent, 'host frames were lost while ffmpeg ran'
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['consumed'] == 0, 'ffmpeg was handed the protocol stdin'
    assert '-nostdin' in report['argv']
