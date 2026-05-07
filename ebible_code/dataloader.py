"""
dataloader.py — eBible corpus loader and filter utility.

Deployed to the top level of the HuggingFace dataset repo alongside the
parquet files so users can run it locally after cloning the dataset.

Usage:
    python dataloader.py filter [--repo REPO] [filter-args]
    python dataloader.py load   [--repo REPO] [filter-args] [load-args]
    python dataloader.py split  [--repo REPO] [filter-args] [load-args] --splits FILE --output-dir DIR
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_REPO = "DavidCBaines/ebible_corpus"
DEFAULT_METADATA_COLUMNS = [
    "translationId",
    "languageCode",
    "countryCode",
    "continentCode",
    "Redistributable",
]
OPERATORS = {"is", "contains", "not", "in"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    column: str
    operator: str
    values: list[str]


@dataclass
class CustomFilterSpec:
    file: str
    join_on: str


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def parse_filter_tokens(tokens: list[str]) -> FilterSpec:
    """Parse [COLUMN, [OPERATOR], VALUE...] into a FilterSpec."""
    if len(tokens) < 2:
        raise ValueError(
            f"--filter requires COLUMN and at least one VALUE, got: {tokens!r}"
        )
    col = tokens[0]
    if tokens[1] in OPERATORS:
        op = tokens[1]
        values = tokens[2:]
        if not values:
            raise ValueError(f"--filter {col} {op} requires at least one value")
    else:
        op = "is"
        values = tokens[1:]
    return FilterSpec(column=col, operator=op, values=values)


def apply_filter(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Apply a single FilterSpec to df; raises ValueError for unknown columns."""
    if spec.column not in df.columns:
        raise ValueError(
            f"Unknown filter column: {spec.column!r}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    col = df[spec.column].astype(str)
    if spec.operator == "is":
        mask = col == spec.values[0]
    elif spec.operator == "contains":
        mask = col.str.contains(spec.values[0], na=False, regex=False)
    elif spec.operator == "not":
        mask = col != spec.values[0]
    elif spec.operator == "in":
        mask = col.isin(spec.values)
    else:
        raise ValueError(f"Unknown operator: {spec.operator!r}")
    return df[mask].reset_index(drop=True)


def apply_filters(df: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    """Apply a sequence of FilterSpecs; AND-combined."""
    for spec in specs:
        df = apply_filter(df, spec)
    return df


def apply_custom_filter(df: pd.DataFrame, spec: CustomFilterSpec) -> pd.DataFrame:
    """Inner-join a user CSV onto df on spec.join_on; adds new columns from the CSV."""
    custom_df = pd.read_csv(spec.file, dtype=str, keep_default_na=False)
    if spec.join_on not in custom_df.columns:
        raise ValueError(
            f"Join column {spec.join_on!r} not found in {spec.file}. "
            f"Available: {sorted(custom_df.columns.tolist())}"
        )
    if spec.join_on not in df.columns:
        raise ValueError(
            f"Join column {spec.join_on!r} not found in metadata. "
            f"Available: {sorted(df.columns.tolist())}"
        )
    merged = df.merge(custom_df, on=spec.join_on, how="inner", suffixes=("", "_custom"))
    drop_cols = [c for c in merged.columns if c.endswith("_custom")]
    return merged.drop(columns=drop_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Repo resolution and loading
# ---------------------------------------------------------------------------

def resolve_repo(repo_arg: Optional[str]) -> str:
    """
    Returns a path string suitable for pd.read_parquet:
      - local directory path (if it exists)
      - hf://datasets/<repo_id>  (HuggingFace URI)
    Falls back to checking cwd for parquet files, then the default HF repo.
    """
    if repo_arg is not None:
        p = Path(repo_arg)
        if p.is_dir():
            return str(p)
        return f"hf://datasets/{repo_arg}"
    cwd = Path(".")
    if (cwd / "main.parquet").exists() and (cwd / "metadata.parquet").exists():
        return str(cwd)
    return f"hf://datasets/{DEFAULT_REPO}"


def _parquet_path(repo: str, filename: str) -> str:
    """Build a full path or hf:// URI for a parquet file in repo."""
    if repo.startswith("hf://"):
        return f"{repo}/{filename}"
    return str(Path(repo) / filename)


def load_metadata_df(repo: str) -> pd.DataFrame:
    return pd.read_parquet(_parquet_path(repo, "metadata.parquet"))


def load_text_df(repo: str, translation_ids: list[str]) -> pd.DataFrame:
    cols = ["vref"] + translation_ids
    df = pd.read_parquet(_parquet_path(repo, "main.parquet"), columns=cols)
    return df.fillna("")


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def build_text_table(main_df: pd.DataFrame, translation_ids: list[str]) -> pd.DataFrame:
    """Select vref + requested translation columns; replace NaN with empty string."""
    available = [tid for tid in translation_ids if tid in main_df.columns]
    return main_df[["vref"] + available].copy().fillna("")


def build_metadata_table(
    metadata_df: pd.DataFrame,
    translation_ids: list[str],
    columns: list[str],
) -> pd.DataFrame:
    """Return metadata rows for the given translations, with specified columns."""
    available_cols = [c for c in columns if c in metadata_df.columns]
    mask = metadata_df["translationId"].isin(translation_ids)
    return metadata_df.loc[mask, available_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def _parse_vref_columns(vref_series: pd.Series) -> pd.DataFrame:
    """Parse 'GEN 1:1' into a DataFrame with string columns book, chapter, verse."""
    extracted = vref_series.str.extract(r'^(\w+)\s+(\d+):(\d+)$')
    extracted.columns = pd.Index(["book", "chapter", "verse"])
    return extracted


def translations_excluded_by_filter(
    splits_df: pd.DataFrame, translation_ids: list[str]
) -> set[str]:
    """Return translationIds present in splits_df but absent from translation_ids."""
    return set(splits_df["translationId"].unique()) - set(translation_ids)


def compute_split_masks(
    vref_series: pd.Series,
    splits_df: pd.DataFrame,
    translation_ids: list[str],
) -> dict[str, dict[str, pd.Series]]:
    """
    Returns {split_name: {translation_id: boolean_mask}}.
    True = verse belongs to this split for this translation.
    Translations not assigned in splits_df default to all-False.
    """
    parsed = _parse_vref_columns(vref_series)
    result: dict[str, dict[str, pd.Series]] = {}

    for split_name in splits_df["split"].unique():
        split_group = splits_df[splits_df["split"] == split_name]
        tid_masks: dict[str, pd.Series] = {}

        for tid in translation_ids:
            tid_rows = split_group[split_group["translationId"] == tid]
            mask = pd.Series(False, index=vref_series.index)
            for _, row in tid_rows.iterrows():
                row_mask = pd.Series(True, index=vref_series.index)
                book = str(row.get("book", "")).strip()
                chapter = str(row.get("chapter", "")).strip()
                verse = str(row.get("verse", "")).strip()
                if book:
                    row_mask &= parsed["book"] == book
                if chapter:
                    row_mask &= parsed["chapter"] == chapter
                if verse:
                    row_mask &= parsed["verse"] == verse
                mask |= row_mask
            tid_masks[tid] = mask

        result[split_name] = tid_masks

    return result


def apply_split(
    text_df: pd.DataFrame,
    split_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Return a copy of text_df with verses outside the split zeroed to ''."""
    df = text_df.copy()
    for tid, mask in split_masks.items():
        if tid in df.columns:
            df.loc[~mask, tid] = ""
    return df


def parse_splits_csv(path: Path | str) -> pd.DataFrame:
    """Read splits.csv; add missing optional columns (book/chapter/verse) as empty."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"translationId", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"splits.csv missing required columns: {missing}")
    for col in ("book", "chapter", "verse"):
        if col not in df.columns:
            df[col] = ""
    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(
    df: pd.DataFrame,
    output_path: Optional[str],
    output_format: str,
) -> Optional[pd.DataFrame]:
    if output_format == "parquet":
        if not output_path:
            sys.exit("ERROR: --output FILE is required for parquet format")
        df.to_parquet(output_path, index=False)
    elif output_format == "huggingface":
        try:
            from datasets import Dataset  # noqa: PLC0415
        except ImportError:
            sys.exit(
                "ERROR: 'datasets' package required for huggingface output. "
                "pip install datasets"
            )
        ds = Dataset.from_pandas(df)
        if output_path:
            ds.save_to_disk(output_path)
        return ds
    elif output_format == "pandas":
        return df
    else:  # csv (default)
        if output_path:
            df.to_csv(output_path, index=False)
        else:
            df.to_csv(sys.stdout, index=False)
    return None


def _metadata_output_path(output: Optional[str], output_format: str) -> Optional[str]:
    if output is None:
        return None
    p = Path(output)
    stem = p.stem + "_metadata"
    return str(p.with_name(stem + p.suffix))


# ---------------------------------------------------------------------------
# Common setup: load and filter metadata
# ---------------------------------------------------------------------------

def setup_filtered_metadata(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[pd.DataFrame, list[str], str]:
    repo = resolve_repo(getattr(args, "repo", None))

    try:
        metadata_df = load_metadata_df(repo)
    except Exception as exc:
        parser.error(f"Could not load metadata.parquet from {repo!r}: {exc}")

    # Pair and apply custom filters
    custom_filters = getattr(args, "custom_filter", None) or []
    join_ons = getattr(args, "join_on", None) or []
    if len(custom_filters) != len(join_ons):
        parser.error(
            f"Each --custom_filter must have a corresponding --join-on "
            f"(got {len(custom_filters)} --custom_filter, {len(join_ons)} --join-on)"
        )
    for cf_file, jo in zip(custom_filters, join_ons):
        try:
            metadata_df = apply_custom_filter(
                metadata_df, CustomFilterSpec(file=cf_file, join_on=jo)
            )
        except ValueError as exc:
            parser.error(str(exc))

    # Parse and apply column filters
    filter_specs: list[FilterSpec] = []
    for tokens in getattr(args, "filter", None) or []:
        try:
            filter_specs.append(parse_filter_tokens(tokens))
        except ValueError as exc:
            parser.error(str(exc))

    try:
        metadata_df = apply_filters(metadata_df, filter_specs)
    except ValueError as exc:
        parser.error(str(exc))

    translation_ids = sorted(metadata_df["translationId"].tolist())
    return metadata_df, translation_ids, repo


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_filter(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    metadata_df, translation_ids, _ = setup_filtered_metadata(args, parser)
    for tid in translation_ids:
        print(tid)
    print(f"\n{len(translation_ids)} translations matched.", file=sys.stderr)


def cmd_load(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    metadata_df, translation_ids, repo = setup_filtered_metadata(args, parser)

    if not translation_ids:
        print("No translations matched the filters.", file=sys.stderr)
        return

    main_df = load_text_df(repo, translation_ids)
    text_table = build_text_table(main_df, translation_ids)

    metadata_columns = args.metadata_columns or DEFAULT_METADATA_COLUMNS
    metadata_columns = [c for c in metadata_columns if c in metadata_df.columns]

    output_format = args.output_format
    write_output(text_table, args.output, output_format)

    if not args.no_metadata:
        metadata_table = build_metadata_table(
            metadata_df, translation_ids, metadata_columns
        )
        meta_out = args.metadata_output or _metadata_output_path(args.output, output_format)
        write_output(metadata_table, meta_out, output_format)

    non_empty_counts = {
        tid: (text_table[tid] != "").sum()
        for tid in translation_ids
        if tid in text_table.columns
    }
    print(f"{len(translation_ids)} translations loaded.", file=sys.stderr)
    for tid, count in non_empty_counts.items():
        if count == 0:
            print(f"  WARNING: {tid} has 0 non-empty verses", file=sys.stderr)


def cmd_split(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    metadata_df, translation_ids, repo = setup_filtered_metadata(args, parser)

    if not translation_ids:
        print("No translations matched the filters.", file=sys.stderr)
        return

    splits_df = parse_splits_csv(args.splits)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.output_format

    for tid in sorted(translations_excluded_by_filter(splits_df, translation_ids)):
        print(f"WARNING: {tid} in splits.csv but excluded by filters", file=sys.stderr)

    main_df = load_text_df(repo, translation_ids)
    text_table = build_text_table(main_df, translation_ids)

    split_masks_by_name = compute_split_masks(
        text_table["vref"], splits_df, translation_ids
    )

    metadata_columns = args.metadata_columns or DEFAULT_METADATA_COLUMNS
    metadata_columns = [c for c in metadata_columns if c in metadata_df.columns]
    ext = ".parquet" if output_format == "parquet" else ".csv"

    for split_name, tid_masks in split_masks_by_name.items():
        split_text = apply_split(text_table, tid_masks)
        write_output(split_text, str(output_dir / f"{split_name}{ext}"), output_format)

        if not args.no_metadata:
            metadata_table = build_metadata_table(
                metadata_df, translation_ids, metadata_columns
            )
            write_output(
                metadata_table,
                str(output_dir / f"{split_name}_metadata{ext}"),
                output_format,
            )

    print(
        f"{len(translation_ids)} translations, {len(split_masks_by_name)} splits.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    filter_parent = argparse.ArgumentParser(add_help=False)
    filter_parent.add_argument(
        "--repo",
        default=None,
        metavar="REPO",
        help="Local directory path or HuggingFace dataset ID (default: DavidCBaines/ebible_corpus)",
    )
    filter_parent.add_argument(
        "--filter",
        nargs="+",
        action="append",
        metavar="TOKEN",
        help="COLUMN [OPERATOR] VALUE... repeatable, AND-combined. "
             "Operators: is (default), contains, not, in",
    )
    filter_parent.add_argument(
        "--custom_filter",
        action="append",
        metavar="FILE",
        help="CSV to join onto metadata; must be paired with --join-on",
    )
    filter_parent.add_argument(
        "--join-on",
        action="append",
        metavar="COLUMN",
        help="Join column for the preceding --custom_filter",
    )

    load_parent = argparse.ArgumentParser(add_help=False)
    load_parent.add_argument("--output", default=None, metavar="FILE")
    load_parent.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "parquet", "huggingface", "pandas"],
        dest="output_format",
    )
    load_parent.add_argument("--metadata-output", default=None, metavar="FILE", dest="metadata_output")
    load_parent.add_argument("--no-metadata", action="store_true", dest="no_metadata")
    load_parent.add_argument(
        "--metadata-columns",
        nargs="+",
        metavar="COL",
        dest="metadata_columns",
        help="Columns to include in the metadata table",
    )

    main_parser = argparse.ArgumentParser(
        prog="dataloader.py",
        description="eBible corpus loader, filter, and split utility.",
    )
    sub = main_parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser(
        "filter",
        parents=[filter_parent],
        help="List matching translationIds and count",
    )
    sub.add_parser(
        "load",
        parents=[filter_parent, load_parent],
        help="Write verse text table and metadata table",
    )
    split_p = sub.add_parser(
        "split",
        parents=[filter_parent, load_parent],
        help="Write per-split verse text and metadata tables",
    )
    split_p.add_argument(
        "--splits",
        required=True,
        metavar="FILE",
        help="Path to splits.csv (columns: translationId, book, chapter, verse, split)",
    )
    split_p.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        dest="output_dir",
        help="Directory for split output files (default: .)",
    )

    return main_parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {"filter": cmd_filter, "load": cmd_load, "split": cmd_split}
    dispatch[args.subcommand](args, parser)


if __name__ == "__main__":
    main()
