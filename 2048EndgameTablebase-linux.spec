# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path


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


hiddenimports = collect_submodules("webview")
hiddenimports += [
    "backend.ai_batch",
    "gi",
    "gi._gi",
    "gi._gi_cairo",
    "native_core.ai_core",
    "native_core.mover_core",
    "native_core.formation_core",
    "webview.platforms.gtk",
]

required_native_binaries = [
    (
        str(require_single_file(native_dir, "ai_core*.so", "ai_core extension")),
        "native_core",
    ),
    (
        str(require_single_file(native_dir, "mover_core*.so", "mover_core extension")),
        "native_core",
    ),
    (
        str(require_single_file(native_dir, "formation_core*.so", "formation_core extension")),
        "native_core",
    ),
    (str(require_file(native_dir / "bookgen_native.so", "bookgen native library")), "native_core"),
    (
        str(require_file(release_native_dir / "bc_family_generation_full", "BC generation helper")),
        "native_core",
    ),
    (
        str(require_file(release_native_dir / "bc_family_solve_full", "BC solve helper")),
        "native_core",
    ),
]

a = Analysis(
    ["backend_server.py"],
    pathex=[],
    binaries=required_native_binaries,
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
