"""Generate assets/language_country_continent.csv by scraping ebible.org/Scriptures.

Reads:
  assets/country_continent.csv  — committed static reference (CountryCode, ContinentCode)

Writes:
  assets/language_country_continent.csv  — translationId, countryCode, continentCode

Run once; re-run only when ebible.org data changes.
"""

import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

SCRIPTURES_URL = "https://ebible.org/Scriptures/"
ASSETS_DIR = Path(__file__).parent.parent / "assets"
CONTINENT_CSV = ASSETS_DIR / "country_continent.csv"
OUTPUT_CSV = ASSETS_DIR / "language_country_continent.csv"


def fetch_scriptures_table(url: str) -> list[tuple[str, str]]:
    """Fetch the eBible Scriptures page and return (translationId, countryCode) pairs."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    # The page has two tables: a nav header and the main data table.
    # Find the first table that contains country.php?c= links (the data table).
    tables = soup.find_all("table")
    table = next(
        (t for t in tables if t.find("a", href=lambda h: h and "country.php?c=" in h)),
        None,
    )
    if table is None:
        raise ValueError(f"No data table with country.php?c= links found at {url}")

    rows = table.find_all("tr")
    pairs = []
    skipped = 0

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue  # header or empty row

        # Territory cell (col 0): href="country.php?c=AL"
        territory_link = cells[0].find("a", href=lambda h: h and "country.php?c=" in h)
        if territory_link is None:
            skipped += 1
            continue
        country_code = territory_link["href"].split("country.php?c=")[-1].strip().upper()

        # Language cell (col 1): href="details.php?id=engPEV"
        language_link = cells[1].find("a", href=lambda h: h and "details.php?id=" in h)
        if language_link is None:
            skipped += 1
            continue
        translation_id = language_link["href"].split("details.php?id=")[-1].strip()

        if country_code and translation_id:
            pairs.append((translation_id, country_code))

    print(f"Scraped {len(pairs)} (translationId, countryCode) pairs; skipped {skipped} rows.",
          file=sys.stderr)
    return pairs


def load_continent_map(csv_path: Path) -> dict[str, str]:
    """Return dict mapping countryCode -> continentCode (first occurrence wins).

    keep_default_na=False prevents pandas from converting 'NA' (Namibia's ISO code)
    or 'NA' (North America continent code) to NaN.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    result = {}
    for _, row in df.iterrows():
        code = str(row["CountryCode"]).strip().upper()
        continent = str(row["ContinentCode"]).strip().upper()
        if code and code not in result:
            result[code] = continent
    return result


def build_mapping(
    pairs: list[tuple[str, str]],
    continent_map: dict[str, str],
) -> pd.DataFrame:
    """Join pairs with continent_map; warn for unresolved country codes."""
    rows = []
    missing_countries = set()

    for translation_id, country_code in pairs:
        continent_code = continent_map.get(country_code)
        if continent_code is None:
            missing_countries.add(country_code)
            continent_code = ""
        rows.append({
            "translationId": translation_id,
            "countryCode": country_code,
            "continentCode": continent_code,
        })

    if missing_countries:
        print(
            f"Warning: {len(missing_countries)} countryCode(s) not found in continent map: "
            f"{sorted(missing_countries)}",
            file=sys.stderr,
        )

    return pd.DataFrame(rows, columns=["translationId", "countryCode", "continentCode"])


def main() -> None:
    if not CONTINENT_CSV.exists():
        print(f"Error: continent mapping not found at {CONTINENT_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching {SCRIPTURES_URL} ...", file=sys.stderr)
    pairs = fetch_scriptures_table(SCRIPTURES_URL)

    continent_map = load_continent_map(CONTINENT_CSV)
    df = build_mapping(pairs, continent_map)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Written {len(df)} rows to {OUTPUT_CSV}", file=sys.stderr)

    # Basic validation
    invalid_country = df[~df["countryCode"].str.match(r"^[A-Z]{2}$", na=False)]
    if not invalid_country.empty:
        print(
            f"Warning: {len(invalid_country)} rows have unexpected countryCode values",
            file=sys.stderr,
        )
    valid_continents = {"AF", "AN", "AS", "EU", "NA", "OC", "SA"}
    invalid_continent = df[~df["continentCode"].isin(valid_continents)]
    if not invalid_continent.empty:
        print(
            f"Warning: {len(invalid_continent)} rows have unexpected continentCode values: "
            f"{df[~df['continentCode'].isin(valid_continents)]['continentCode'].unique().tolist()}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
