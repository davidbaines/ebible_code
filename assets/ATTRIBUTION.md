# Data Attribution

This directory contains reference data files derived from the following sources.

---

## Glottolog 5.3

**File**: `glottolog_families.csv`

**Licence**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

**Citation**:
> Hammarström, Harald & Forkel, Robert & Haspelmath, Martin & Bank, Sebastian. 2024.
> *Glottolog 5.3*. Leipzig: Max Planck Institute for Evolutionary Anthropology.
> Available at https://glottolog.org

**Source URL**: https://cdstar.eva.mpg.de/bitstreams/EAEA0-608B-9919-A962-0/glottolog_languoid.csv.zip

`glottolog_families.csv` is derived from Glottolog's `languoid.csv` by extracting the
`iso639P3code`, `id` (glottocode), `name`, `family_id`, and `parent_id` fields and
computing the top-level `family_name` and full ancestor-path `classification` for each
language-level entry. The derivation script is `ebible_code/get_glottolog_families.py`.

---

## Country-Continent Mapping

**File**: `country_continent.csv`

**Source**: GitHub Gist by Steve Withington
https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c

`country_continent.csv` was derived from the Gist CSV by retaining only the
`CountryCode` and `ContinentCode` columns.

---

## eBible.org Scriptures Table

**File**: `language_country_continent.csv`

**Source**: https://ebible.org/Scriptures/

`language_country_continent.csv` is derived by scraping the HTML table at
https://ebible.org/Scriptures/ and extracting `translationId` and `countryCode` from
the `details.php?id=` and `country.php?c=` href patterns. The derivation script is
`ebible_code/generate_language_country_continent.py`.

Individual Bible translations in the eBible corpus are governed by their own licences
(public domain, Creative Commons, etc.) as recorded in `ebible_status.csv`.
