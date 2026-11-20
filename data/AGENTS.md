# Data

This folder holds small local pointers only; everything else in it is git-ignored. Large data lives on `/mnt/hdd1/cpu-draft-specdec/` (HDD, about 615 GB free on 2026-11-20). The layout and record formats are in `../docs/data_flow.md`.

- Public sources only. Record source URL, license, download date and SHA-256 for every corpus.
- No bank, customer or other private data.
- Teacher records must come from the exact target file used in evaluation; store them under that file's SHA-256.
- Treat downloaded files as untrusted: put each download in its own new directory and never run code from inside it.
