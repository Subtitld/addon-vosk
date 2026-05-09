# Subtitld Vosk add-on

Offline speech-to-text for [Subtitld](https://subtitld.org), powered by
[Vosk](https://alphacephei.com/vosk/) Kaldi recognisers. CPU-only, lightweight
(~80 MB binary, plus 40 MB–2 GB per language model), works without an
internet connection once the model is installed.

## What this is

Subtitld's transcription engine is implemented as a **separate add-on** that
talks to the host over a JSON-line stdio protocol. Users install it from
*Add-ons → Browse* (1-click) or via the *Install from file…* button after
downloading a release zip from this repository.

Why a separate process?

- Keeps the Subtitld binary lean — Vosk + the smallest English model alone
  is ~120 MB.
- Lets advanced users swap to a larger Vosk model, or replace Vosk entirely
  with another ASR add-on (Whisper, Faster-Whisper, …) without rebuilding
  Subtitld.
- Sandboxes the recogniser: a Kaldi crash kills the add-on subprocess, not
  the editor.

## Supported languages

| Language        | Default model                        | Size  |
|-----------------|--------------------------------------|-------|
| English (US)    | `vosk-model-small-en-us-0.15`        | 40 MB |
| Portuguese      | `vosk-model-small-pt-0.3`            | 31 MB |
| Spanish         | `vosk-model-small-es-0.42`           | 39 MB |
| French          | `vosk-model-small-fr-0.22`           | 41 MB |
| German          | `vosk-model-small-de-0.15`           | 45 MB |
| Italian         | `vosk-model-small-it-0.22`           | 48 MB |
| Russian         | `vosk-model-small-ru-0.22`           | 45 MB |
| Chinese (CN)    | `vosk-model-small-cn-0.22`           | 42 MB |
| Japanese        | `vosk-model-small-ja-0.22`           | 48 MB |

Larger high-accuracy variants are listed in `manifest.json` under `models[]`
and can be selected from the add-on's Configure dialog. See the full
[catalogue at alphacephei.com](https://alphacephei.com/vosk/models) for
languages not in the list above — drop a feature request issue if you want
one added.

Models are downloaded on demand from `alphacephei.com` to
`<user-data>/subtitld/addons/org.subtitld.vosk/models/<model-id>/` the first
time you transcribe in a given language. Disable `auto_download` in the
Configure panel to require manual placement instead.

## Building from source

You'll need Python 3.10+ and the host platform's PyInstaller prerequisites.

```bash
# 1. Install dependencies into a venv.
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e '.[build]'

# 2. Build the redistributable bundle.
pyinstaller pyinstaller.spec --noconfirm

# 3. Pack into the release zip Subtitld expects.
PLATFORM=$(python -c 'import platform, sys; print({"linux":"linux","darwin":"macos","win32":"windows"}[sys.platform])')
ARCH=$(python -c 'import platform; print({"x86_64":"x86_64","AMD64":"x86_64","arm64":"arm64","aarch64":"arm64"}[platform.machine()])')
mkdir -p release-staging
cp -r dist/vosk-addon release-staging/
cp manifest.json LICENSE README.md release-staging/vosk-addon/
( cd release-staging && zip -r "../org.subtitld.vosk-1.0.0-${PLATFORM}-${ARCH}.zip" vosk-addon )
```

The resulting `org.subtitld.vosk-<version>-<platform>-<arch>.zip` is the
shape Subtitld's *Install from file…* expects.

### Running directly (development)

```bash
python -m vosk_addon
```

Then drive it manually by piping JSON-line frames in. A minimal smoke test:

```bash
printf '%s\n' \
  '{"type":"ready","host":"smoke","host_version":"0.0","protocol":1}' \
  '{"id":"r1","type":"asr.transcribe","params":{"audio_path":"/tmp/sample.wav","language":"en-us","options":{}}}' \
| python -m vosk_addon
```

## Protocol

The add-on speaks the Subtitld add-on protocol v1 — see
[`subtitld/modules/addons/protocol.py`](https://github.com/Subtitld/subtitld/blob/main/src/subtitld/modules/addons/protocol.py)
for the spec. Briefly, it:

- emits `hello` on startup with `addon`, `version`, `protocol`, and a
  `capabilities` array that advertises `task: "asr.transcribe"`;
- waits for the host's `ready` ack before processing requests;
- handles `asr.transcribe` requests with `{audio_path, language, options}`;
- streams one `partial` frame per finalised segment
  (`{start, end, text, speaker}`);
- emits one terminal `result` (`{segments, duration_sec, model}`) or `error`
  per request id;
- honours `cancel` (best-effort, takes effect at the next AcceptWaveform
  checkpoint — typically <1s) and `shutdown` (graceful, drains workers).

## Layout

```
.
├── manifest.json                  # advertised to the host on install
├── pyproject.toml                 # `pip install -e .` for development
├── pyinstaller.spec               # builds dist/vosk-addon/
├── vosk_addon/
│   ├── __init__.py
│   └── __main__.py                # protocol loop + worker
├── tests/
│   └── test_protocol_loop.py
└── .github/workflows/release.yml  # builds + uploads zip on tag push
```

## Licence

Apache 2.0 — see [LICENSE](LICENSE). The underlying Vosk Kaldi runtime is
also Apache 2.0; individual model checkpoints carry their own licences (most
are Apache 2.0, the Portuguese full-size model is GPL-3.0). Each model's
licence is recorded in `manifest.json:models[*].license`.
