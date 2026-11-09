# FILEDB ownership on the reorg branch

QDP++ no longer includes the FILEDB submodule, database wrapper headers,
--enable-filedb option, libraries or exported FILEDB link flags. It does not
gain a datalib dependency. Chroma obtains FILEDB from installed datalib and
retains its old database adapter locally while legacy writers are modernized.
QDP's XML and lattice I/O/communication behavior is unchanged.

The macOS ARM library/examples build succeeded. Full make check still fails
in unchanged upstream x86 SIMD tests, reproduced on the previous installation.
See datalib's docs/reorg/FILEDB-TOOLS-REGRESSIONS.md and recorded evidence.
