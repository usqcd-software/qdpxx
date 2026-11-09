# FILEDB ownership on the reorg branch

QDP++ no longer includes the FILEDB submodule, database wrapper headers,
--enable-filedb option, libraries or exported FILEDB link flags. It does not
gain a datalib dependency. Chroma obtains FILEDB from installed datalib and
uses datalib storage classes directly for the legacy writers. Chroma's temporary
database adapter and serializable wrappers have been removed.
QDP's XML and lattice I/O/communication behavior is unchanged.

The macOS ARM library/examples build succeeded. Full make check still fails
in unchanged upstream x86 SIMD tests, reproduced on the previous installation.
See datalib's docs/reorg/FILEDB-TOOLS-REGRESSIONS.md and recorded evidence.
