"""
Download Glottolog 5.3 languoid.csv.zip and produce assets/glottolog_families.csv.

Output columns: languageCode, glottocode, family_name, classification
  - family_name: top-level ancestor name; "Isolate" for languages whose family_id is empty
  - classification: slash-separated path from root to language (inclusive)

Usage:
    poetry run python ebible_code/get_glottolog_families.py
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

GLOTTOLOG_URL = (
    "https://cdstar.eva.mpg.de/bitstreams/EAEA0-608B-9919-A962-0/"
    "glottolog_languoid.csv.zip"
)

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_PATH = ASSETS_DIR / "glottolog_families.csv"


def load_macrolanguage_overrides(path: Path) -> dict[str, str]:
    """Return {ebible_language_code: glottolog_lookup_code}; empty dict if file absent."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Drop blank rows (e.g. header-only file)
    df = df[df["ebible_language_code"] != ""]
    return dict(zip(df["ebible_language_code"], df["glottolog_lookup_code"]))


def load_languoids(url: str) -> pd.DataFrame:
    """Download Glottolog zip and return languoid.csv as a DataFrame."""
    print(f"Downloading Glottolog data from {url} ...", file=sys.stderr)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open("languoid.csv") as f:
            return pd.read_csv(f, dtype=str, keep_default_na=False)


def build_family_records(
    df: pd.DataFrame, overrides: dict[str, str]
) -> list[dict]:
    """
    For each language-level row with a non-empty iso639P3code compute
    (languageCode, glottocode, family_name, classification) and return as a list
    of dicts.  First occurrence wins when two rows share an ISO code.
    """
    id_to_row = df.set_index("id")
    langs_with_iso = df[(df["level"] == "language") & (df["iso639P3code"] != "")]

    seen_iso: set[str] = set()
    records: list[dict] = []

    for _, row in langs_with_iso.iterrows():
        iso = row["iso639P3code"]
        if iso in seen_iso:
            continue
        seen_iso.add(iso)

        # Apply macrolanguage override: redirect the lookup to a specific code
        if iso in overrides:
            lookup_iso = overrides[iso]
            override_rows = df[
                (df["iso639P3code"] == lookup_iso) & (df["level"] == "language")
            ]
            if override_rows.empty:
                print(
                    f"WARNING: macrolanguage override {iso} -> {lookup_iso} "
                    "not found in Glottolog; using original row",
                    file=sys.stderr,
                )
                data_row = row
            else:
                data_row = override_rows.iloc[0]
        else:
            data_row = row

        glottocode = data_row["id"]

        # Trace ancestor chain upward via parent_id
        path_names = [data_row["name"]]
        current = data_row
        while current["parent_id"] and current["parent_id"] in id_to_row.index:
            current = id_to_row.loc[current["parent_id"]]
            path_names.append(current["name"])
        path_names.reverse()  # now top-down: root → ... → language

        # family_name from family_id; empty family_id means the language IS the root (isolate)
        family_id = data_row["family_id"]
        if not family_id:
            family_name = "Isolate"
        elif family_id in id_to_row.index:
            family_name = id_to_row.loc[family_id]["name"]
        else:
            family_name = "Unknown"

        classification = "/".join(path_names)
        records.append(
            {
                "languageCode": iso,
                "glottocode": glottocode,
                "family_name": family_name,
                "classification": classification,
            }
        )

    return records


def write_glottolog_families(records: list[dict], output_path: Path) -> None:
    df_out = pd.DataFrame(
        records, columns=["languageCode", "glottocode", "family_name", "classification"]
    )
    df_out.to_csv(output_path, index=False)
    print(f"Wrote {len(df_out)} rows to {output_path}", file=sys.stderr)


def main() -> None:
    overrides_path = ASSETS_DIR / "macrolanguage_overrides.csv"
    overrides = load_macrolanguage_overrides(overrides_path)
    if overrides:
        print(f"Loaded {len(overrides)} macrolanguage override(s)", file=sys.stderr)

    df = load_languoids(GLOTTOLOG_URL)
    records = build_family_records(df, overrides)
    write_glottolog_families(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()
