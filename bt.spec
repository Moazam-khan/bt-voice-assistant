# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for BT.

Build with: pyinstaller bt.spec --noconfirm

Two data-collection fixes were required by testing (see commit history
for the full diagnosis): silero_vad and chromadb both load resources
dynamically (importlib.resources / string-based plugin imports) in ways
PyInstaller's static analysis can't see on its own, causing the packaged
app to silently crash on a background thread. --collect-data/--collect-all
force those files and hidden imports to be bundled.

config/, models/, and logs/ are NOT bundled here — they're copied next to
the built .exe as external, editable files (see README's packaging
section), matching how bt_core/config.py resolves paths for a frozen app.
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [("bt_core/ui/chat.html", "bt_core/ui")]
binaries = []
hiddenimports = []

datas += collect_data_files("silero_vad")

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
