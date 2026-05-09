# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld Vosk add-on.

Produces a `dist/vosk-addon/` directory containing the launcher binary
(`vosk-addon` on POSIX, `vosk-addon.exe` on Windows) plus the bundled
libvosk runtime, shipped Python deps, and the addon manifest.

Run from the addon root:

    pyinstaller pyinstaller.spec --noconfirm

The release workflow then zips `dist/vosk-addon/` together with
`manifest.json`, `LICENSE`, and `README.md` into the platform-tagged
archive consumed by Subtitld's AddonsDialog (`Install from file…`).
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
import sys
from pathlib import Path

# `pyinstaller pyinstaller.spec` runs the spec from CWD; resolve the addon
# root from the spec's own location so `pyinstaller` can be invoked from
# anywhere.
SPEC_ROOT = Path(SPECPATH).resolve()


def _vosk_runtime_binaries() -> list[tuple[str, str]]:
    """Locate the libvosk shared library bundled with the `vosk` Python
    package and re-emit it as a PyInstaller `binaries` entry.

    PyInstaller's ctypes hook usually picks this up automatically, but on
    Linux + manylinux wheels the .so sits next to `vosk/__init__.py` rather
    than in a recognised place — adding it here is cheap insurance against
    silent runtime failures.
    """
    try:
        import vosk  # type: ignore
    except ImportError:
        return []
    pkg_dir = Path(vosk.__file__).parent
    out: list[tuple[str, str]] = []
    for pattern in ('*.so', '*.dylib', '*.dll'):
        for path in pkg_dir.glob(pattern):
            out.append((str(path), 'vosk'))
    return out


a = Analysis(
    [str(SPEC_ROOT / 'vosk_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=_vosk_runtime_binaries(),
    datas=[],
    hiddenimports=[
        'vosk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # The addon doesn't use Qt or Tk; exclude them so the bundle stays
        # small (~10 MB instead of ~80 MB on Linux).
        'PySide6', 'PyQt6', 'PyQt5',
        'tkinter', 'Tkinter', '_tkinter',
        'matplotlib', 'numpy.testing', 'scipy', 'pandas',
    ],
    noarchive=False,
)

# Strip the bundled libexpat. PyInstaller copies whatever expat is on the
# build host; on modern Ubuntu (22.04+) `libpython3.12` is linked against
# expat ≥ 2.6 which exposes `XML_SetReparseDeferralEnabled`. CI runners
# (and many user systems) have an older expat available, so the bundled
# .so doesn't have that symbol → libpython fails to load with
#
#     undefined symbol: XML_SetReparseDeferralEnabled
#
# unless the user happens to LD_PRELOAD a newer copy. Removing our bundled
# expat lets the dynamic linker pick the system one, which is virtually
# guaranteed to be present (linux-vdso for libc, expat ships with every
# distro). On macOS / Windows the binaries list won't contain libexpat so
# this filter is a no-op.
def _drop_bundled_libexpat(entries):
    keep = []
    for entry in entries:
        # Each entry is (dest_name, src_path, type_code).
        name = entry[0] if entry else ''
        base = os.path.basename(name)
        if base.startswith('libexpat.so') or base.startswith('libexpat.dylib'):
            continue
        keep.append(entry)
    return keep


a.binaries = _drop_bundled_libexpat(a.binaries)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vosk-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # libvosk doesn't survive UPX on some platforms.
    console=True,  # Communication is over stdio — no GUI window.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='vosk-addon',
)
