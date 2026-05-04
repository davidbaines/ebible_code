"""Upload, download, and tag HuggingFace datasets from the command line.

Usage:
    python huggingface.py upload  <repo_id> <folder> [--files F [F ...]] [--tags T [T ...]] [--token TOKEN]
    python huggingface.py download <repo_id> <folder> [--tag TAG] [--token TOKEN]
    python huggingface.py tag      <repo_id> (--add-tags T [T ...] | --list-tags) [--token TOKEN]
"""

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, HfFolder, snapshot_download
from packaging.version import InvalidVersion, Version

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    from huggingface_hub.utils import RepositoryNotFoundError


# ── Auth ──────────────────────────────────────────────────────────────────────

_AUTH_HELP = (
    "No HuggingFace token found. Provide one using one of these methods:\n"
    "  1. Pass --token TOKEN on the command line\n"
    "  2. Set HF_TOKEN=<your_token> in a .env file in the current directory\n"
    "  3. Run:  huggingface-cli login\n"
    "     (installs via: pip install huggingface_hub)"
)


def resolve_token(cli_token=None):
    """Return the first available HF token, or None if none found.

    Resolution order: CLI arg → .env HF_TOKEN → hf auth login cache.
    """
    if cli_token:
        return cli_token

    load_dotenv()
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    try:
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    return None


def require_token(cli_token=None):
    """Return a token or print help and exit with code 1."""
    token = resolve_token(cli_token)
    if not token:
        print(f"Error: {_AUTH_HELP}", file=sys.stderr)
        sys.exit(1)
    return token


# ── Version tag utilities ─────────────────────────────────────────────────────

def is_version_tag(tag: str) -> bool:
    """True if tag starts with 'v' or 'V' (indicates a semantic version)."""
    return tag.startswith(("v", "V"))


def parse_version_tag(tag: str):
    """Strip leading v/V and parse as packaging.Version. Returns None if unparseable."""
    try:
        return Version(tag.lstrip("vV"))
    except InvalidVersion:
        return None


def check_version_tags(new_tags, existing_tags, input_fn=input):
    """Warn and confirm if any new version tag is <= the highest existing version tag.

    Returns True if safe to proceed, False if the user declines.
    """
    existing_versions = []
    for tag in existing_tags:
        if not is_version_tag(tag):
            continue
        v = parse_version_tag(tag)
        if v is None:
            print(f"Warning: could not parse existing tag '{tag}' as a version.", file=sys.stderr)
        else:
            existing_versions.append((tag, v))

    if not existing_versions:
        return True

    max_tag, max_ver = max(existing_versions, key=lambda x: x[1])

    for tag in new_tags:
        if not is_version_tag(tag):
            continue
        new_ver = parse_version_tag(tag)
        if new_ver is None:
            print(f"Warning: could not parse new tag '{tag}' as a version — skipping version check.",
                  file=sys.stderr)
            continue
        if new_ver <= max_ver:
            answer = input_fn(
                f"Warning: '{tag}' ({new_ver}) is the same or older than existing tag "
                f"'{max_tag}' ({max_ver}). Continue? [y/N]: "
            )
            if answer.strip().lower() != "y":
                return False

    return True


def _get_existing_tags(api: HfApi, repo_id: str) -> list:
    """Return list of tag name strings for the given dataset repo."""
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    return [t.name for t in refs.tags]


# ── Upload ────────────────────────────────────────────────────────────────────

_UPLOAD_EXTENSIONS = {".parquet", ".csv", ".md"}


def upload_dataset(repo_id, folder, files=None, tags=None, token=None, input_fn=input):
    """Upload files from a local folder to a HuggingFace dataset repo.

    Args:
        repo_id:  HuggingFace repo ID, e.g. "DavidCBaines/ebible_corpus".
        folder:   Local folder path containing files to upload.
        files:    Optional list of filenames (relative to folder) to upload.
                  If None, all .parquet/.csv/.md files in folder are used.
        tags:     Optional list of tag strings to apply after upload.
        token:    HuggingFace API token.
        input_fn: Callable used for interactive prompts (injectable for tests).

    Returns:
        True on success, False if the user aborted.
    """
    api = HfApi(token=token)
    folder = Path(folder)

    # 1. Repo existence check
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        answer = input_fn(
            f"Repository '{repo_id}' does not exist on HuggingFace.\n"
            f"Create it now? [y/N]: "
        )
        if answer.strip().lower() != "y":
            print("Aborted. Check the repository name and try again.")
            return False
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Created repository: https://huggingface.co/datasets/{repo_id}")

    # 2. Determine upload paths
    if files:
        upload_paths = []
        for name in files:
            p = folder / name
            if not p.exists():
                print(f"Error: specified file not found: {p}", file=sys.stderr)
                sys.exit(1)
            upload_paths.append(p)
    else:
        upload_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _UPLOAD_EXTENSIONS
        )

    if not upload_paths:
        print("No files found to upload.")
        return False

    # 3. Confirm file list with user
    print("\nFiles to upload:")
    for p in upload_paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<40}  {size_kb:>8.1f} KB")

    answer = input_fn(f"\nUpload {len(upload_paths)} file(s) to '{repo_id}'? [y/N]: ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        return False

    # 4. Version tag regression check
    if tags:
        new_version_tags = [t for t in tags if is_version_tag(t)]
        if new_version_tags:
            existing = _get_existing_tags(api, repo_id)
            if not check_version_tags(new_version_tags, existing, input_fn=input_fn):
                print("Aborted.")
                return False

    # 5. Upload all files in a single commit
    operations = [
        CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p))
        for p in upload_paths
    ]
    names = ", ".join(p.name for p in upload_paths)
    print(f"\nUploading {len(upload_paths)} file(s) in one commit: {names} ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload {len(upload_paths)} file(s)",
    )
    print("done")

    # 6. Apply tags
    if tags:
        print()
        for tag in tags:
            api.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")
            print(f"  Tagged: {tag}")

    print(f"\nUpload complete: https://huggingface.co/datasets/{repo_id}")
    return True


# ── Download ──────────────────────────────────────────────────────────────────

def download_dataset(repo_id, folder, tag=None, token=None):
    """Download a HuggingFace dataset repo to a local folder.

    Args:
        repo_id: HuggingFace repo ID.
        folder:  Local directory to download into (created if absent).
        tag:     Tag/revision to download. Defaults to 'main'.
        token:   HuggingFace API token (optional for public repos).

    Returns:
        Path to the local download directory.
    """
    revision = tag if tag else "main"
    dest = Path(folder)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{repo_id}' at revision '{revision}' → '{dest}' ...")

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        token=token,
    )

    local_path = Path(local_dir)

    # Report Parquet files
    parquet_files = sorted(local_path.glob("*.parquet"))
    if parquet_files:
        print("\nParquet file summary:")
        for pf in parquet_files:
            _report_parquet(pf)

    # Overall summary
    all_files = [p for p in local_path.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in all_files)
    print(f"\nDownload complete: {len(all_files)} file(s), {total_size / 1_048_576:.2f} MB total")

    return local_path


def _report_parquet(path: Path):
    """Print row/column summary for a single Parquet file."""
    meta = pq.read_metadata(str(path))
    schema = pq.read_schema(str(path))

    num_rows = meta.num_rows
    col_names = schema.names
    num_cols = len(col_names)

    print(f"  {path.name}")
    print(f"    Rows:    {num_rows:,}")
    print(f"    Columns: {num_cols}")

    if col_names:
        preview = col_names[:10]
        suffix = " ..." if num_cols > 10 else ""
        print(f"    Column names: {', '.join(preview)}{suffix}")

    if meta.num_row_groups > 1:
        print(f"    Row groups: {meta.num_row_groups}")


# ── Tag ───────────────────────────────────────────────────────────────────────

def list_tags(repo_id, token=None):
    """Print all tags on a HuggingFace dataset repo, sorted alphabetically.

    Returns the sorted list of tag name strings.
    """
    api = HfApi(token=token)
    tags = sorted(_get_existing_tags(api, repo_id))
    if not tags:
        print(f"No tags found on '{repo_id}'.")
    else:
        print("\n".join(tags))
    return tags


def add_tags(repo_id, tags, token=None, input_fn=input):
    """Add one or more tags to a HuggingFace dataset repo at its current HEAD.

    Returns True on success, False if the user aborted.
    """
    api = HfApi(token=token)

    new_version_tags = [t for t in tags if is_version_tag(t)]
    if new_version_tags:
        existing = _get_existing_tags(api, repo_id)
        if not check_version_tags(new_version_tags, existing, input_fn=input_fn):
            print("Aborted.")
            return False

    for tag in tags:
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")
        print(f"Tagged: {tag}")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="huggingface",
        description="Upload, download, and tag HuggingFace datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s upload DavidCBaines/ebible_corpus ./huggingface --tags v2.0\n"
            "  %(prog)s download DavidCBaines/ebible_corpus ./local_copy --tag v1.0\n"
            "  %(prog)s tag DavidCBaines/ebible_corpus --list-tags\n"
            "  %(prog)s tag DavidCBaines/ebible_corpus --add-tags v2.0 latest\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # upload
    up = sub.add_parser("upload", help="Upload files from a local folder to a dataset repo")
    up.add_argument("repo_id", help="HuggingFace repo ID, e.g. Owner/dataset-name")
    up.add_argument("folder", help="Local folder containing files to upload")
    up.add_argument(
        "--files", nargs="+", metavar="FILE",
        help="Specific filenames to upload (default: all .parquet, .csv, .md in folder)",
    )
    up.add_argument(
        "--tags", nargs="+", metavar="TAG",
        help="One or more tags to apply after a successful upload",
    )
    up.add_argument("--token", metavar="TOKEN", help="HuggingFace API token")

    # download
    dl = sub.add_parser("download", help="Download a dataset repo to a local folder")
    dl.add_argument("repo_id", help="HuggingFace repo ID")
    dl.add_argument("folder", help="Local folder to download into (created if absent)")
    dl.add_argument(
        "--tag", metavar="TAG",
        help="Tag or revision to download (default: main)",
    )
    dl.add_argument("--token", metavar="TOKEN", help="HuggingFace API token")

    # tag
    tg = sub.add_parser("tag", help="List or add tags on a dataset repo")
    tg.add_argument("repo_id", help="HuggingFace repo ID")
    tg.add_argument("--token", metavar="TOKEN", help="HuggingFace API token")
    grp = tg.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--add-tags", nargs="+", metavar="TAG",
        help="One or more tags to add to the repo",
    )
    grp.add_argument(
        "--list-tags", action="store_true",
        help="List all tags on the repo, sorted alphabetically",
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "upload":
        token = require_token(args.token)
        ok = upload_dataset(
            repo_id=args.repo_id,
            folder=args.folder,
            files=args.files,
            tags=args.tags,
            token=token,
        )
        sys.exit(0 if ok else 1)

    elif args.command == "download":
        token = resolve_token(args.token)
        download_dataset(
            repo_id=args.repo_id,
            folder=args.folder,
            tag=args.tag,
            token=token,
        )

    elif args.command == "tag":
        if args.list_tags:
            token = resolve_token(args.token)
            list_tags(repo_id=args.repo_id, token=token)
        else:
            token = require_token(args.token)
            ok = add_tags(
                repo_id=args.repo_id,
                tags=args.add_tags,
                token=token,
            )
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
