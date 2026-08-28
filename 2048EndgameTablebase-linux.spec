# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path


hiddenimports = collect_submodules("webview")
hiddenimports += [
    "gi",
    "gi._gi",
    "gi._gi_cairo",
    "native_core.ai_core",
    "native_core.mover_core",
    "native_core.formation_core",
    "webview.platforms.gtk",
]

optional_native_binaries = []
for helper_name in ("bc_family_generation_full", "bc_family_solve_full"):
    for helper_path in (
        Path("native_core") / helper_name,
        Path("native_core/build-formation") / helper_name,
    ):
        if helper_path.exists():
            optional_native_binaries.append((str(helper_path), "native_core"))
            break

a = Analysis(
    ["backend_server.py"],
    pathex=[],
    binaries=[
        ("native_core/bookgen_native.so", "native_core"),
        *optional_native_binaries,
    ],
    datas=[
        ("docs_and_configs/default_patterns.json", "docs_and_configs"),
        ("docs_and_configs/patterns_config.json", "docs_and_configs"),
        ("docs_and_configs/performance_evaluations.json", "docs_and_configs"),
        ("docs_and_configs/themes.json", "docs_and_configs"),
        ("docs_and_configs/runtime_deletion_threshold.txt", "docs_and_configs"),
        ("docs_and_configs/help", "docs_and_configs/help"),
        ("pic", "pic"),
        ("font", "font"),
        ("favicon.ico", "."),
        ("mathjax", "mathjax"),
        ("frontend/dist", "frontend/dist"),
    ],
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
    name="2048EndgameTablebase",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["pic/2048_2.ico"],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="2048EndgameTablebase",
)
