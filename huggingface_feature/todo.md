# huggingface.py — To-Do

## Setup
- [x] Add dependencies to `pyproject.toml`: `huggingface_hub`, `pyarrow`, `python-dotenv`, `packaging`
- [x] Create `ebible_code/huggingface.py` skeleton (argparse + subparsers, importable functions)

## Authentication module
- [x] Implement `resolve_token(cli_token)` — tries CLI arg → `.env` HF_TOKEN → cached login → error message
- [x] Test: `--token` arg used when provided
- [x] Test: `.env` HF_TOKEN read when no CLI token
- [x] Test: cached `hf auth login` token used as fallback
- [x] Test: all three fail → error message names all three options

## `upload` command
- [x] Implement repo existence check (`HfApi.repo_info`); prompt to create if missing
- [x] Test: missing repo → user confirms → repo created
- [x] Test: missing repo → user declines → clean exit
- [x] Implement file listing: auto-discover `.parquet`, `.csv`, `.md` in folder (non-recursive)
- [x] Implement `--files` override: resolve named files relative to folder; error if any missing
- [x] Test: auto-discovery returns only correct extensions
- [x] Test: `--files` uses only specified files
- [x] Test: `--files` with missing file → error
- [x] Implement interactive file confirmation (print list with sizes; prompt)
- [x] Test: user declines → clean exit
- [x] Implement version tag warning: fetch existing tags, compare semantically, prompt if new ≤ existing
- [x] Test: new tag `v1.0` when `v2.0` exists → warning shown
- [x] Test: new tag `v3.0` when `v2.0` exists → no warning
- [x] Test: user declines tag warning → clean exit
- [x] Implement file upload loop (`HfApi.upload_file` per file, with progress)
- [x] Test: `upload_file` called once per confirmed file
- [x] Implement post-upload tagging (`HfApi.create_tag` per tag)
- [x] Test: `create_tag` called for each tag after upload

## `download` command
- [x] Implement `snapshot_download` call with `revision` defaulting to `"main"`
- [x] Test: `--tag v1.0` → `revision="v1.0"` passed to snapshot_download
- [x] Test: no `--tag` → `revision="main"`
- [x] Implement Parquet reporting: read metadata with `pyarrow.parquet.read_metadata`
- [x] Report rows, columns, column names, row group count (if > 1)
- [x] Test: real `.parquet` file → correct rows/cols/names printed
- [x] Implement download summary: total files, total size on disk

## `tag` command
- [x] Implement `--list-tags`: fetch all tags, print alphabetically
- [x] Test: tags returned out of order → output is sorted
- [x] Test: no tags on repo → message printed
- [x] Implement `--add-tags`: version warning (same logic as upload), then `create_tag` per tag
- [x] Test: version warning fires correctly
- [x] Test: `create_tag` called for each tag
- [x] Enforce mutual exclusivity of `--add-tags` and `--list-tags` in argparse
- [x] Test: both flags together → argparse error

## Version comparison utility
- [x] Implement `parse_version_tag(tag)` → `packaging.version.Version` or `None`
- [x] Implement `is_version_tag(tag)` → bool (starts with v/V)
- [x] Implement `check_version_tags(new_tags, existing_tags)` → warns and prompts if regression
- [x] Unit tests: `v1.0 < v2.0`, `v1.0 == v1.0`, `v2.0 > v1.9`, `v1.10 > v1.9`, unparseable tag skipped with warning

## CLI / argparse
- [x] Test: `upload --help` exits cleanly
- [x] Test: `download --help` exits cleanly
- [x] Test: `tag --help` exits cleanly
- [x] Test: unknown subcommand → helpful error

## Importable API
- [x] Verify `from ebible_code.huggingface import upload_dataset, download_dataset, add_tags, list_tags` works
