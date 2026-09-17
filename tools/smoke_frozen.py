"""Smoke-test a frozen build the way a user's machine runs it.

    python tools/smoke_frozen.py dist/vosk-addon/vosk-addon

Two checks against the real binary, over the real protocol:

  * **TLS.** SSL_CERT_FILE and SSL_CERT_DIR point at paths that do not
    exist, so the bundled OpenSSL finds no system CA store — what happens on
    macOS, Fedora/Arch and the Flatpak runtime, where the build machine's CA
    paths are absent. Without this, the CI runner's own CA store would hide
    the problem. The model download from alphacephei.com must still work;
    the request is cancelled once the first megabyte has arrived.
  * **ffmpeg environment (Linux).** The frozen bootloader points
    LD_LIBRARY_PATH at the bundle's `_internal`; a system ffmpeg must not
    inherit that. A stand-in ffmpeg records what it was given.

POSIX only: the stand-in is a shell script, and on Windows Python reads the
system certificate store directly, so the env vars can't simulate the
failure there anyway.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REQ_ID = 'smoke'
MODEL_ID = 'vosk-model-small-en-us-0.15'  # the `en` default, ~40 MB

# Records LD_LIBRARY_PATH (or its absence), then writes the output file the
# add-on expects — the last argument, as with the real ffmpeg.
_FAKE_FFMPEG = '''#!/bin/sh
if [ "${LD_LIBRARY_PATH+set}" = set ]; then
    printf 'set:%s' "$LD_LIBRARY_PATH" > "$SMOKE_FFMPEG_REPORT"
else
    printf 'unset' > "$SMOKE_FFMPEG_REPORT"
fi
for last; do :; done
printf 'RIFF smoke' > "$last"
'''


class Addon:
    def __init__(self, binary: Path, env: dict[str, str]):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env)
        self.frames: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self.frames.put(json.loads(line))
        self.frames.put(None)

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr.append(line.decode('utf-8', errors='replace').rstrip())

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def next_frame(self, timeout: float) -> dict:
        try:
            frame = self.frames.get(timeout=timeout)
        except queue.Empty:
            self.fail(f'no frame within {timeout:.0f}s')
        if frame is None:
            self.fail('add-on exited early')
        return frame

    def transcribe(self, params: dict, *, cancel_once_downloading: bool,
                   timeout: float = 300.0) -> dict:
        """Handshake, run one request, return its terminal frame."""
        hello = self.next_frame(60)
        if hello.get('type') != 'hello':
            self.fail(f'expected hello, got {hello}')
        self.send({'type': 'ready', 'host': 'smoke', 'host_version': '0'})
        self.send({'type': 'asr.transcribe', 'id': REQ_ID, 'params': params})
        cancelled = False
        while True:
            frame = self.next_frame(timeout)
            if frame.get('id') != REQ_ID:
                continue
            if frame.get('type') in ('error', 'result'):
                return frame
            # A non-zero download fraction means bytes arrived over TLS.
            if (cancel_once_downloading and not cancelled
                    and frame.get('type') == 'progress' and frame.get('value', 0) > 0):
                print(f'  download under way: {frame.get("message")}')
                self.send({'type': 'cancel', 'target': REQ_ID})
                cancelled = True

    def close(self) -> None:
        try:
            self.send({'type': 'shutdown'})
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()

    def fail(self, why: str):
        self.proc.kill()
        tail = '\n'.join(self.stderr[-30:])
        raise SystemExit(f'FAIL: {why}\n--- add-on stderr ---\n{tail}')


def run(binary: Path, work: Path, data_dir: Path, *, host_ld_path: str | None,
        cancel_once_downloading: bool) -> tuple[dict, str]:
    """One add-on process, one request. Returns (terminal frame, ffmpeg report)."""
    report = work / 'ffmpeg-report.txt'
    report.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SUBTITLD_VOSK_', 'LD_LIBRARY_PATH'))}
    env.update({
        'SSL_CERT_FILE': str(work / 'no-such-ca-file.pem'),
        'SSL_CERT_DIR': str(work / 'no-such-ca-dir'),
        'SUBTITLD_VOSK_DATA_DIR': str(data_dir),
        'SUBTITLD_FFMPEG_EXECUTABLE': str(work / 'ffmpeg'),
        'SMOKE_FFMPEG_REPORT': str(report),
    })
    if host_ld_path is not None:
        env['LD_LIBRARY_PATH'] = host_ld_path

    addon = Addon(binary, env)
    try:
        frame = addon.transcribe(
            {'audio_path': str(work / 'in.mp3'), 'language': 'en'},
            cancel_once_downloading=cancel_once_downloading)
    finally:
        addon.close()
    got = report.read_text() if report.exists() else '<ffmpeg never ran>'
    print(f'  terminal frame: {frame}')
    print(f'  ffmpeg saw LD_LIBRARY_PATH: {got}')
    return frame, got


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if os.name != 'posix':
        print('skipped: POSIX only (see module docstring)')
        return 0
    binary = Path(sys.argv[1]).resolve()
    bundle_libs = str(binary.parent / '_internal')
    failures = []

    with tempfile.TemporaryDirectory(prefix='vosk-smoke-') as tmp:
        work = Path(tmp)
        (work / 'in.mp3').write_bytes(b'ID3 smoke')
        fake = work / 'ffmpeg'
        fake.write_text(_FAKE_FFMPEG)
        fake.chmod(0o755)

        print('TLS with no system CA store (model download):')
        frame, _ = run(binary, work, work / 'empty-data', host_ld_path=None,
                       cancel_once_downloading=True)
        if frame.get('type') == 'error' and (
                frame.get('code') == 'network_unavailable'
                or 'certificate' in str(frame.get('message', '')).lower()):
            failures.append(f'TLS: the model download failed: {frame.get("message")}')

        if sys.platform.startswith('linux'):
            # A placeholder model skips the download, so ffmpeg runs at once.
            # Loading it fails afterwards; only the ffmpeg report matters here.
            seeded = work / 'seeded-data'
            (seeded / 'models' / MODEL_ID).mkdir(parents=True)
            (seeded / 'models' / MODEL_ID / 'placeholder').write_text('smoke')
            host_libs = str(work / 'host-libs')
            for host_ld_path, expected in ((None, 'unset'), (host_libs, f'set:{host_libs}')):
                print(f'ffmpeg environment, LD_LIBRARY_PATH={host_ld_path}:')
                _, seen = run(binary, work, seeded, host_ld_path=host_ld_path,
                              cancel_once_downloading=False)
                if seen != expected:
                    failures.append(f'ffmpeg env: expected {expected!r}, got {seen!r}')
                if bundle_libs in seen:
                    failures.append(f'ffmpeg env: LD_LIBRARY_PATH still includes {bundle_libs}')

    for failure in failures:
        print(f'FAIL: {failure}')
    if not failures:
        print('OK')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
