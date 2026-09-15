"""Generate the web manual from the source branch; requires Pandoc."""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
ref = sys.argv[1] if len(sys.argv) > 1 else "master"
source = subprocess.check_output(["git", "show", f"{ref}:docs/manual.tex"], cwd=root)
with tempfile.TemporaryDirectory() as temporary:
    tex = Path(temporary) / "manual.tex"
    tex.write_bytes(source)
    result = subprocess.run([
        "pandoc", str(tex), "--from=latex", "--to=html5", "--standalone",
        "--toc", "--mathml", "--fail-if-warnings",
        "--lua-filter=" + str(root / "scripts/manual-math.lua"),
        "--metadata", "title=QDP++ Reference Manual",
        "--css=stylesheets/manual.css",
    ], check=True, capture_output=True, text=True)
    html = result.stdout.replace('<body>', '<body>\n<nav class="manual-nav" aria-label="Manual navigation"><a href="./">QDP++ home</a> · <a href="manual.pdf">Manual PDF</a> · <a href="usr/html/index.html">User guide</a></nav>')
    (root / "manual.html").write_text(html)
