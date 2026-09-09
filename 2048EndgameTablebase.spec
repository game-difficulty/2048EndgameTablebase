# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import importlib.util


repo_root = Path(SPECPATH)
native_dir = repo_root / "native_core"
release_native_dir = native_dir / "build-formation"


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing required {label}: {path}")
    return path


def require_single_file(directory, pattern, label):
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "none"
        raise SystemExit(
            f"Expected exactly one {label} matching {pattern} in {directory}; found: {rendered}"
        )
    return matches[0]


required_native_binaries = [
    (
        str(require_single_file(native_dir, "ai_core*.pyd", "ai_core extension")),
        "native_core",
    ),
    (
        str(require_single_file(native_dir, "mover_core*.pyd", "mover_core extension")),
        "native_core",
    ),
    (
        str(require_single_file(native_dir, "formation_core*.pyd", "formation_core extension")),
        "native_core",
    ),
    (str(require_file(native_dir / "libgcc_s_seh-1.dll", "MinGW runtime")), "native_core"),
    (str(require_file(native_dir / "libgomp-1.dll", "OpenMP runtime")), "native_core"),
    (str(require_file(native_dir / "libwinpthread-1.dll", "MinGW threading runtime")), "native_core"),
    (str(require_file(native_dir / "bookgen_native.dll", "bookgen native library")), "native_core"),
    (
        str(require_file(release_native_dir / "bc_family_generation_full.exe", "BC generation helper")),
        "native_core",
    ),
    (
        str(require_file(release_native_dir / "bc_family_solve_full.exe", "BC solve helper")),
        "native_core",
    ),
]

canonical_runtime_dlls = {
    name.lower(): str(require_file(native_dir / name, "MinGW runtime"))
    for name in ("libgcc_s_seh-1.dll", "libgomp-1.dll", "libwinpthread-1.dll")
}

optional_native_binaries = []
for helper_path in [native_dir / "share_pinhole_helper.exe", *native_dir.glob("miniupnpc-*.dll")]:
    if helper_path.exists():
        optional_native_binaries.append((str(helper_path), "native_core"))
if not any(Path(path).name.startswith("miniupnpc-") for path, _ in optional_native_binaries):
    miniupnpc_spec = importlib.util.find_spec("miniupnpc")
    if miniupnpc_spec and miniupnpc_spec.origin:
        for dll_path in Path(miniupnpc_spec.origin).resolve().parent.glob("miniupnpc-*.dll"):
            optional_native_binaries.append((str(dll_path), "native_core"))

a = Analysis(
    ['backend_server.py'],
    pathex=[],
    binaries=[*required_native_binaries, *optional_native_binaries],
    datas=[('docs_and_configs/default_patterns.json', 'docs_and_configs'), ('docs_and_configs/patterns_config.json', 'docs_and_configs'), ('docs_and_configs/performance_evaluations.json', 'docs_and_configs'), ('docs_and_configs/runtime_deletion_threshold.txt', 'docs_and_configs'), ('docs_and_configs/themes.json', 'docs_and_configs'), ('docs_and_configs/help', 'docs_and_configs/help'), ('pic', 'pic'), ('favicon.ico', '.'), ('mathjax', 'mathjax'), ('frontend/dist', 'frontend/dist'), ('7zip/7z.dll', '.'), ('7zip/7z.exe', '.')],
    hiddenimports=['backend.ai_batch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# PyInstaller may discover another MinGW runtime through the build environment.
# Keep root-level dependency copies identical to the native_core release copies.
a.binaries = [
    (
        destination,
        canonical_runtime_dlls.get(Path(destination).name.lower(), source)
        if Path(destination).parent == Path(".")
        else source,
        typecode,
    )
    for destination, source, typecode in a.binaries
]
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
