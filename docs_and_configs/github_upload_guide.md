# GitHub Upload Guide: 2048 Endgame Tablebase

This document outlines the recommended file structure and inclusion list for open-sourcing the **2048 Endgame Tablebase** project to GitHub. The goal is to provide a clean repository that contains all necessary source code while excluding local configuration, large data files, and build artifacts.

## 1. Summary of Items to Upload

### Core Source Code
- `backend_server.py`: The main entry point for the FastAPI server.
- `Config.py`, `SignalHub.py`, `error_bridge.py`: Essential application-level utilities.
- `egtb_core/`: The logic core (Algorithm modules, AI engines, and table generation logic).
- `backend/`: API handlers, session management, and game services.
- `minigames/`: Compatibility layers and assets for specific game variants.

### Frontend
- `frontend/`: The Vue 3 source code.
  - **Include**: `src/`, `public/`, `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`.
  - **Exclude**: `node_modules/`, `dist/`.

### Documentation & Templates
- `docs_and_configs/`: 
  - **Include**: `help/`, `code_structure.txt`, and **template** versions of `patterns_config.json` and `default_patterns.json`.
  - **Exclude**: `config`, `mistakes_book.pkl`, `color_schemes.txt`, and any `*.error` files.
- `README.md`: (To be created) Project introduction, installation guide, and usage manual.
- `LICENSE`: (To be created) Choose an appropriate open-source license (e.g., MIT, GPL-3.0).

### Assets
- `pic/`, `font/`: Static assets required for UI rendering.
- `favicon.ico`: Application icon.

---

## 2. Recommended Repository Structure

```text
2048EndgameTablebase/
├── .gitignore               # Critical for keeping the repo clean
├── LICENSE                  # Legal permissions
├── README.md                # Project documentation
├── backend_server.py        # Entrypoint
├── Config.py                # Config management
├── SignalHub.py             # Event/Signal bus
├── error_bridge.py          # Backend-Frontend error channel
├── egtb_core/               # Core algorithms (Calculator, Search, etc.)
│   ├── __init__.py
│   ├── AIPlayer.py
│   ├── BoardMover.py
│   └── ...
├── backend/                 # API handlers and business logic
├── minigames/               # Game variants and compatibility
├── frontend/                # Vue 3 Frontend
│   ├── src/
│   ├── package.json
│   └── ...
├── docs_and_configs/        # Internal docs and help files
│   ├── help/                # Markdown help files (EN/ZH)
│   ├── code_structure.txt   # Updated structure overview
│   ├── patterns_config.json # Keep as template/default
│   └── default_patterns.json
├── pic/                     # UI Assets
├── font/                    # Typography
└── tools/                   # Standalone testing scripts
```

---

## 3. Recommended .gitignore

Create a `.gitignore` file in the root directory to prevent accidental upload of sensitive or unnecessary files.

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.venv/
venv/

# Local Data & Logs
logger.txt
docs_and_configs/config
docs_and_configs/mistakes_book.pkl
docs_and_configs/color_schemes.txt
docs_and_configs/*.error
*.log

# C++ / Native Build
AIPlayer_cpp/build/
AIPlayer_cpp/wasm_project/build/
.vscode/
.idea/

# Node.js / Frontend
frontend/node_modules/
frontend/dist/
frontend/.env.local

# Large Tablebase Data (User Specific)
# Usually tables are stored in a path provided by the user in settings.
# Exclude any local table data folder if it exists in root.
TablebaseData/
*.book
*.z
```

---

## 4. Special Considerations

### Large Files
If you intend to host common tablebases (e.g., small 512 endgame tables), consider using **GitHub Releases** or **Git LFS** instead of direct commits.

### Configuration Templates
Since `docs_and_configs/config` contains local state and `mistakes_book.pkl` contains personal progress, **do not upload them**. Ensure `Config.py` correctly generates a default configuration when the file is missing.

### Third-party Dependencies
- **MathJax**: If you ship a local copy in `mathjax/`, verify the license. Alternatively, switch to a CDN in the frontend to reduce repo size.
