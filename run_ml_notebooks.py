"""Execute code cells in a generated notebook for reproducibility checks."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from IPython.display import display


def main(notebook_path):
    path = Path(notebook_path).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": display}
    old_cwd = Path.cwd()
    os.chdir(path.parent)
    try:
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            print(f"[{path.name}] code cell {index}", flush=True)
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main(sys.argv[1])
