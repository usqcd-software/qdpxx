# GitHub Pages

The public site is https://usqcd-software.github.io/qdpxx/.
In GitHub **Settings → Pages**, choose **Deploy from a branch**, branch
**gh-pages**, folder **/ (root)**. Save if the setting needs changing.

The files must be committed to `gh-pages`; untracked local files are not published.
The homepage uses these paths relative to the branch root:

- `manual.pdf`: existing PDF manual
- `manual.html`: generated HTML reference manual
- `usr/html/index.html`: existing Doxygen user guide, with its supporting files

GitHub `blob`, `tree`, and raw-content URLs are not website URLs. Relative links
also preserve the `/qdpxx/` project prefix. `.nojekyll` serves the generated
HTML and supporting files without Jekyll processing.

Review the changes, commit only the intended site files, and push `gh-pages`.
Check the Pages deployment in GitHub Actions before checking the public URLs.
The local untracked `docs/` and `other_libs/` directories are not needed for this update.

## Regenerating the HTML manual

The manual was generated from `docs/manual.tex` on the local `master` branch
using Pandoc with a table of contents and native MathML. The existing PDF was
preserved. The helper below reads the source without switching branches:

```sh
python3 scripts/build-manual.py
```

Requires Python 3 and Pandoc. Supply another source ref with
`python3 scripts/build-manual.py origin/master` after fetching updates.
