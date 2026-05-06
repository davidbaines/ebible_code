---
language:
- multilingual
license: other
multilinguality: multilingual
size_categories:
- 10K<n<100K
task_categories:
- translation
- text-generation
pretty_name: eBible Parallel Corpus
---

# eBible Parallel Corpus

A verse-aligned parallel corpus of {{TRANSLATION_COUNT}} Bible translations across {{LANGUAGE_COUNT}} languages, prepared from publicly redistributable texts hosted by [eBible.org](https://ebible.org).

Generated: {{GENERATED_DATE}}

## Dataset Description

Each row corresponds to one verse reference from the standard 41,899-verse reference list (`vref.txt`). Every translation occupies one column, identified by its `translationId`. Rows are positionally aligned: row N in every translation column refers to the same verse.

- **Verse count:** {{VERSE_COUNT}}
- **Translations:** {{TRANSLATION_COUNT}}
- **Languages:** {{LANGUAGE_COUNT}}

## Dataset Structure

### Data Files

| File | Description |
|---|---|
| `main.parquet` | Wide-format parallel corpus: one row per verse, one column per translation |
| `metadata.parquet` | Per-translation metadata: language, licence, coverage, versification |

### Fields in `main.parquet`

| Column | Type | Description |
|---|---|---|
| `vref` | string | Verse reference, e.g. `GEN 1:1` |
| `<translationId>` | string | Translated text for that verse; `""` = untranslated; `<range>` = verse merged into preceding range |

### Fields in `metadata.parquet`

| Column | Description |
|---|---|
| `translationId` | Unique identifier matching column names in `main.parquet` |
| `languageCode` | ISO 639-3 language code |
| `languageName` | Language name in the vernacular |
| `languageNameInEnglish` | Language name in English |
| `dialect` | Dialect, if applicable |
| `title` | Full title of the translation |
| `shortTitle` | Abbreviated title |
| `description` | Description of the translation |
| `textDirection` | `ltr` or `rtl` |
| `script` | Writing script (e.g. `Latin`, `Arabic`) |
| `inScript` | Whether the text uses the language's native script |
| `OTbooks` / `OTchapters` / `OTverses` | Old Testament coverage |
| `NTbooks` / `NTchapters` / `NTverses` | New Testament coverage |
| `DCbooks` / `DCchapters` / `DCverses` | Deuterocanonical coverage |
| `inferred_versification` | Versification number inferred for this translation |
| `licence_Licence_Type` | Licence type (e.g. `Creative Commons`) |
| `licence_Licence_Version` | Licence version (e.g. `4.0`) |
| `licence_CC_Licence_Link` | URL to the CC licence deed |
| `licence_Copyright_Holder` | Copyright holder |
| `licence_Copyright_Years` | Copyright year(s) |
| `licence_Translation_by` | Translating organisation or individual |
| `licence_Vernacular_Title` | Title as stated in the licence file |
| `Copyright` | Copyright statement from eBible.org |
| `UpdateDate` | Date last updated on eBible.org |
| `sourceDate` | Date of the source text |
| `publicationURL` | Publication page on eBible.org |

## Loading the Dataset

The dataset ships with `dataloader.py`, a CLI utility for filtering, loading, and splitting the corpus without writing any Python.

### Installation

```bash
pip install pandas pyarrow huggingface_hub
```

### Basic usage

**List matching translations (no output files written):**

```bash
# All redistributable translations in Europe or Asia
python dataloader.py filter \
  --filter Redistributable True \
  --filter continentCode in EU AS

# Germanic-family languages (requires glottolog_families.csv)
python dataloader.py filter \
  --custom_filter assets/glottolog_families.csv --join-on languageCode \
  --filter family_name contains Germanic
```

**Load verse text and metadata to CSV:**

```bash
python dataloader.py load \
  --filter Redistributable True \
  --filter continentCode EU \
  --output eu_verses.csv \
  --metadata-columns translationId languageCode countryCode Redistributable
```

This writes two files: `eu_verses.csv` (one row per verse, one column per translation) and `eu_verses_metadata.csv` (one row per translation).

**Write train/test/val splits:**

```bash
python dataloader.py split \
  --filter Redistributable True \
  --splits my_splits.csv \
  --output-dir splits/
```

`splits.csv` columns: `translationId`, `book`, `chapter`, `verse`, `split`. Omit trailing columns to assign larger scopes — a row with only `translationId` and `split` assigns the entire Bible to that split.

**Load against a local copy of the dataset:**

```bash
python dataloader.py filter --repo /path/to/local/dataset --filter languageCode eng
```

### Filter operators

| Operator | Example | Meaning |
|---|---|---|
| *(default)* | `--filter Redistributable True` | Exact match |
| `contains` | `--filter family_name contains Germanic` | Substring match |
| `not` | `--filter translationId not engKJV` | Not equal |
| `in` | `--filter continentCode in EU AS AF` | One of a list |

Multiple `--filter` flags are AND-combined.

### Programmatic use

```python
from datasets import load_dataset

# Load the full dataset directly — no dataloader.py needed
ds = load_dataset("DavidCBaines/ebible_corpus", split="train")
df = ds.to_pandas()
```

## Licence Breakdown

All translations in this dataset are redistributable as confirmed by eBible.org. Individual licence terms vary by translation. For per-translation licence details — including licence type, version, CC link, copyright holder, and translating organisation — load `metadata.parquet`.

## Citation

If you use this dataset in your work, please cite David Baines, SIL, PABNLP:

```
@misc{ebible,
  title  = {eBible Parallel Corpus},
  author = {David Baines SIL Global PABNLP},
  url    = {https://pabnlp.org/},
  year   = {{GENERATED_DATE}}
}
```