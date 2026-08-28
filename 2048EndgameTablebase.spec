# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import glob
import importlib.util


optional_native_binaries = []
for helper_path in [Path("native_core/share_pinhole_helper.exe"), *Path("native_core").glob("miniupnpc-*.dll")]:
    if helper_path.exists():
        optional_native_binaries.append((str(helper_path), "native_core"))
for helper_name in ("bc_family_generation_full.exe", "bc_family_solve_full.exe"):
    for helper_path in (
        Path("native_core") / helper_name,
        Path("native_core/build-bc-release") / helper_name,
        Path("native_core/build-bc-relwithdeb") / helper_name,
        Path("native_core/build-formation") / helper_name,
    ):
        if helper_path.exists():
            optional_native_binaries.append((str(helper_path), "native_core"))
            break
if not any(Path(path).name.startswith("miniupnpc-") for path, _ in optional_native_binaries):
    miniupnpc_spec = importlib.util.find_spec("miniupnpc")
    if miniupnpc_spec and miniupnpc_spec.origin:
        for dll_path in glob.glob(str(Path(miniupnpc_spec.origin).resolve().parent / "miniupnpc-*.dll")):
            optional_native_binaries.append((dll_path, "native_core"))

a = Analysis(
    ['backend_server.py'],
    pathex=[],
    binaries=[('native_core/libgcc_s_seh-1.dll', 'native_core'), ('native_core/libgomp-1.dll', 'native_core'), ('native_core/libwinpthread-1.dll', 'native_core'), ('native_core/bookgen_native.dll', 'native_core'), *optional_native_binaries],
    datas=[('docs_and_configs/default_patterns.json', 'docs_and_configs'), ('docs_and_configs/patterns_config.json', 'docs_and_configs'), ('docs_and_configs/performance_evaluations.json', 'docs_and_configs'), ('docs_and_configs/runtime_deletion_threshold.txt', 'docs_and_configs'), ('docs_and_configs/themes.json', 'docs_and_configs'), ('docs_and_configs/help', 'docs_and_configs/help'), ('pic', 'pic'), ('favicon.ico', '.'), ('mathjax', 'mathjax'), ('frontend/dist', 'frontend/dist'), ('7zip/7z.dll', '.'), ('7zip/7z.exe', '.')],
    hiddenimports=[],
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
    name='2048EndgameTablebase',
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
    icon=['pic\\2048_2.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='2048EndgameTablebase',
)
