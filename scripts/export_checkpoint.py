"""CLI entry point for exporting a checkpoint to a smaller fp16 distributable.

Thin wrapper: all logic lives in `yegpt.export` (pure core + tested), matching the codebase
convention of keeping orchestration in the package and the script surface trivial. Run:

    python scripts/export_checkpoint.py [--checkpoint PATH] [--out PATH]
"""

from yegpt.export import main

if __name__ == "__main__":
    main()
