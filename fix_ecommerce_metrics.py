"""Repair the multiline metric assignment in all e-commerce experiment notebooks."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parent / "project" / "e-commerce-comment-analytis"
for path in sorted(folder.glob("e-commerce-analytics-exp*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    fixed = False
    for cell in notebook["cells"]:
        value = "".join(cell.get("source", []))
        old = '''precision,
recall,
f1 = multiclass_metrics('''
        if old in value:
            value = value.replace(old, '''precision, recall, f1 = multiclass_metrics(''')
            cell["source"] = value.splitlines(True)
            fixed = True
    if not fixed:
        raise RuntimeError(f"Metric assignment not found in {path}")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
