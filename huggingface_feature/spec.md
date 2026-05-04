# huggingface.py — Specification

## Goals

A general-purpose CLI tool and importable Python module for uploading, downloading, and tagging datasets on HuggingFace Hub. Designed to work with any HuggingFace dataset, not just `ebible_corpus`.

---

## CLI Interface

```
python huggingface.py upload <repo_id> <folder>
                             [--files FILE [FILE ...]]
                             [--tags TAG [TAG ...]]
                             [--token TOKEN]

python huggingface.py download <repo_id> <folder>
                               [--tag TAG]
                               [--token TOKEN]

python huggingface.py tag <repo_id>
                          (--add-tags TAG [TAG ...] | --list-tags)
                          [--token TOKEN]
```

All subcommands accept `--help`.

---

## Authentication

Resolved in this order. The first that succeeds is used:

1. `--token TOKEN` passed on the command line
2. `HF_TOKEN` variable in a `.env` file in the working directory (loaded via `python-dotenv`)
3. Token cached by `hf auth login`

If all three fail, the script prints a clear explanation of all three options (including the exact `hf auth login` command) and exits with a non-zero code.

---

## `upload` command

### Signature
```
upload <repo_id> <folder> [--files FILE [FILE ...]] [--tags TAG [TAG ...]] [--token TOKEN]
```

### Behaviour (in order)

1. **Authenticate** using the auth resolution above.

2. **Check repo existence.** Query HuggingFace for `<repo_id>` (type=dataset).
   - If the repo does not exist: print a warning naming the repo, ask the user to confirm creation.
     - If confirmed: create the repo and continue.
     - If declined: exit cleanly with a message (gives the user a chance to fix a typo).

3. **Determine files to upload.**
   - If `--files` is given: use exactly those filenames, resolved relative to `<folder>`. Error if any named file does not exist.
   - Otherwise: find all `.parquet`, `.csv`, and `.md` files directly in `<folder>` (non-recursive).

4. **Confirm files with the user.** Print the list of files (with sizes) and ask for confirmation.
   - If declined: exit cleanly with a message.

5. **Check version tags** (only if `--tags` is given and any tag starts with `v` or `V`).
   - Fetch all existing tags from the repo.
   - Extract existing version tags (those starting with `v`/`V`, case-insensitive).
   - Parse all version tags using semantic versioning (strip leading `v`/`V`, parse as `MAJOR.MINOR[.PATCH]`).
   - For each new version tag, if it is equal to or older than the highest existing version tag: print a warning naming both tags and ask for interactive confirmation.
     - If declined: exit cleanly.

6. **Upload files** to the root of the repo using `huggingface_hub.HfApi.upload_file` for each file. Print progress per file.

7. **Apply tags** (if `--tags` given). Call `HfApi.create_tag` for each tag after a successful upload.

---

## `download` command

### Signature
```
download <repo_id> <folder> [--tag TAG] [--token TOKEN]
```

### Behaviour

1. **Authenticate.**

2. **Download** the repo snapshot at `--tag` (or `main` if omitted) into `<folder>` using `huggingface_hub.snapshot_download`. Print which revision is being fetched.

3. **Report Parquet files.** For each `.parquet` file in the downloaded folder:
   - Open with `pyarrow.parquet.read_metadata` (reads footer only, no full data load).
   - Report: filename, number of rows, number of columns.
   - Report column names if present in the schema.
   - Report row group count if > 1 (indicates a chunked/sharded file).

4. Print a summary: total files downloaded, total size on disk.

---

## `tag` command

### Signature
```
tag <repo_id> (--add-tags TAG [TAG ...] | --list-tags) [--token TOKEN]
```

`--add-tags` and `--list-tags` are mutually exclusive.

### `--list-tags` behaviour

1. Fetch all tags from `<repo_id>` (no auth required for public repos; use token if available).
2. Print all tags, one per line, sorted alphabetically (like `git branch`).
3. If there are no tags, print a message saying so.

### `--add-tags` behaviour

1. **Authenticate.**
2. **Check version tags** using the same logic as `upload` step 5.
3. **Apply tags** to the current `HEAD` of the repo using `HfApi.create_tag` for each tag.
4. Print confirmation of each tag applied.

---

## Version tag comparison

- A tag is a **version tag** if it starts with `v` or `V` (case-insensitive).
- Strip the leading `v`/`V` and parse as `MAJOR.MINOR[.PATCH]` using `packaging.version.Version`.
- If parsing fails (non-numeric), skip it in comparisons and emit a warning that it could not be parsed.
- "Same or older" means `new_version <= max(existing_version_tags)`.

---

## Implementation notes

### Dependencies (add to `pyproject.toml`)
- `huggingface_hub` — Hub API (upload, download, tags, repo management)
- `pyarrow` — Parquet metadata reading
- `python-dotenv` — `.env` loading
- `packaging` — semantic version comparison

### Module structure
```
ebible_code/
    huggingface.py      # CLI entry point + all importable functions
tests/
    test_hf_auth.py
    test_hf_upload.py
    test_hf_download.py
    test_hf_tag.py
    test_hf_cli.py
```

### Importable API
All subcommand logic should be in named functions so the module can be imported:
```python
from ebible_code.huggingface import upload_dataset, download_dataset, add_tags, list_tags
```

### argparse structure
Use `argparse.ArgumentParser` with `add_subparsers`. Each subcommand registers its own arguments. `--add-tags` and `--list-tags` use a `mutually_exclusive_group`.

---

## Verification

Each piece of functionality is verified as follows:

| Feature | How to verify |
|---|---|
| Auth: `--token` | Pass a dummy token; confirm it is used (mock HfApi) |
| Auth: `.env` | Set `HF_TOKEN` in a temp `.env`; confirm it is read |
| Auth: `hf auth login` | Remove token sources; confirm cached token is used |
| Auth: all fail | Remove all token sources; confirm error message lists all three options |
| Upload: repo missing → create | Mock `repo_info` to raise 404; confirm creation prompt fires |
| Upload: repo missing → decline | Decline creation; confirm clean exit |
| Upload: file listing (auto) | Folder with mixed file types; confirm only `.parquet/.csv/.md` listed |
| Upload: file listing (`--files`) | Specify subset; confirm only those files shown |
| Upload: confirmation decline | Decline file list; confirm clean exit |
| Upload: version tag warning | New tag `v1.0` when `v2.0` exists; confirm warning + prompt |
| Upload: version tag warning → decline | Decline; confirm clean exit |
| Upload: files uploaded | Mock `upload_file`; confirm called once per file |
| Upload: tags applied after upload | Mock `create_tag`; confirm called for each tag |
| Download: correct revision | `--tag v1.0`; confirm `snapshot_download` called with `revision="v1.0"` |
| Download: default revision | No `--tag`; confirm `revision="main"` |
| Download: Parquet report | Real `.parquet` file; confirm rows/cols/names printed |
| Tag: `--list-tags` sorted | Tags returned out of order; confirm output is alphabetical |
| Tag: `--add-tags` version warning | Same warning logic as upload |
| Tag: mutually exclusive flags | Both `--add-tags` and `--list-tags`; confirm argparse error |
| Version comparison | Unit tests for `v1.0 < v2.0`, `v1.0 == v1.0`, `v2.0 > v1.9` |
