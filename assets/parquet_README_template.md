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

```python
from datasets import load_dataset

ds = load_dataset("your-org/ebible-parallel", split="train")

# Access a specific translation
kjv = ds["eng-engKJV"]

# Filter to New Testament
nt_start = ds.filter(lambda row: row["vref"].startswith("MAT"))
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