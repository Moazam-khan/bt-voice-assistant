# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for BT.

Build with: pyinstaller bt.spec --noconfirm

Three data-collection fixes were required by testing (see commit history
for the full diagnosis) — silero_vad, chromadb, and piper all load
resources dynamically or from a native (non-Python) library in ways
PyInstaller's static analysis can't see on its own, causing the packaged
app to crash — silently on a background thread for the first two, and as
an uncatchable native crash for piper's bundled espeak-ng C library,
which fails to find its phoneme data and aborts the whole process rather
than raising a Python exception. --collect-data/--collect-all force
those files and hidden imports to be bundled.

config/, models/, and logs/ are NOT bundled here — they're copied next to
the built .exe as external, editable files (see README's packaging
section), matching how bt_core/config.py resolves paths for a frozen app.
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ("bt_core/ui/chat.html", "bt_core/ui"),
    ("bt_core/ui/chat.js", "bt_core/ui"),
    ("bt_core/ui/icon.ico", "bt_core/ui"),
    ("bt_core/ui/css/base.css", "bt_core/ui/css"),
    ("bt_core/ui/css/sidebar.css", "bt_core/ui/css"),
    ("bt_core/ui/css/center_panel.css", "bt_core/ui/css"),
    ("bt_core/ui/css/orb.css", "bt_core/ui/css"),
    ("bt_core/ui/css/chat_panel.css", "bt_core/ui/css"),
]
binaries = []
hiddenimports = []

datas += collect_data_files("silero_vad")
datas += collect_data_files("piper")

chromadb_datas, chromadb_binaries, chromadb_hiddenimports = collect_all("chromadb")
datas += chromadb_datas
binaries += chromadb_binaries
hiddenimports += chromadb_hiddenimports

a = Analysis(
    ["run_bt.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="bt_core/ui/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BT",
)
