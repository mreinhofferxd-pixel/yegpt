"""CLI entry point for exporting a batch of short sample fragments from a checkpoint.

Thin wrapper: all logic lives in `yegpt.export_samples` (pure core + tested), matching the
codebase convention of keeping orchestration in the package and the script surface trivial. Run:

    python scripts/export_samples.py [--checkpoint PATH] [--out PATH] [-n N] [--num-tokens M]
"""

from yegpt.export_samples import main

if __name__ == "__main__":
    main()
