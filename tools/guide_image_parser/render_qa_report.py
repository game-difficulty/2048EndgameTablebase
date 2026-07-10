from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.guide_image_parser.qa_report import main


if __name__ == "__main__":
    main()
