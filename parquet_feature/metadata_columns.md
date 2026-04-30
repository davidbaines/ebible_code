# Metadata Parquet Column Selection

Mark each column `[x]` to include it in `metadata.parquet`, or `[ ]` to exclude it.

Columns are grouped by source. Pipeline-internal columns (paths, dates, hashes) are pre-marked excluded by default.

---

## ORIGINAL_COLUMNS
Sourced from eBible.org's `translations.csv`.

- [x] `languageCode` — ISO 639-3 language code (e.g. `eng`, `zpi`)
- [x] `translationId` — Unique identifier; matches the column names in `main.parquet` (e.g. `eng-engKJV`)
- [x] `languageName` — Language name in the vernacular script
- [x] `languageNameInEnglish` — Language name in English
- [x] `dialect` — Dialect name or description, if applicable
- [x] `homeDomain` — Geographic region or domain where the language is primarily spoken
- [x] `title` — Full title of the translation in the vernacular
- [x] `description` — Longer description of the translation
- [x] `Redistributable` — Whether eBible.org permits redistribution (`True`/`False`)
- [x] `Copyright` — Copyright statement as provided by eBible.org
- [x] `UpdateDate` — Date the translation was last updated on eBible.org
- [x] `publicationURL` — URL to the translation's publication page on eBible.org
- [x] `OTbooks` — Number of Old Testament books present in this translation
- [x] `OTchapters` — Number of Old Testament chapters present
- [x] `OTverses` — Number of Old Testament verses present
- [x] `NTbooks` — Number of New Testament books present
- [x] `NTchapters` — Number of New Testament chapters present
- [x] `NTverses` — Number of New Testament verses present
- [x] `DCbooks` — Number of Deuterocanonical/Apocrypha books present
- [x] `DCchapters` — Number of Deuterocanonical/Apocrypha chapters present
- [x] `DCverses` — Number of Deuterocanonical/Apocrypha verses present
- [x] `FCBHID` — Faith Comes By Hearing identifier for linking to audio Bible recordings
- [x] `Certified` — Whether the translation has been certified (provenance unclear)
- [x] `inScript` — Whether the text uses the native script of the language (`True`/`False`)
- [x] `swordName` — Sword Project module name (used by Bible software like Sword/BibleDesktop)
- [x] `rodCode` — Registry of Dialects code
- [x] `textDirection` — Text direction: `ltr` or `rtl`
- [x] `downloadable` — Whether eBible.org permits direct download (distinct from Redistributable)
- [x] `font` — Recommended font for rendering this translation correctly
- [x] `shortTitle` — Abbreviated title
- [x] `PODISBN` — Print-on-demand ISBN, if available
- [x] `script` — Writing script (e.g. `Latin`, `Arabic`, `Devanagari`)
- [x] `sourceDate` — Date of the source text used as the basis for this translation

---

## LICENCE_COLUMNS
Parsed from `copr.htm` inside each Paratext project folder.

- [ ] `licence_ID` — Internal licence record identifier parsed from the copyright file
- [ ] `licence_File` — Name of the copyright file (typically `copr.htm`)
- [ ] `licence_Language` — Language name as stated in the licence file
- [ ] `licence_Dialect` — Dialect as stated in the licence file
- [x] `licence_Vernacular_Title` — Title in the vernacular as stated in the licence
- [x] `licence_Licence_Type` — Licence type (e.g. `Creative Commons`, `Custom`)
- [x] `licence_Licence_Version` — Licence version (e.g. `4.0` for CC BY 4.0)
- [x] `licence_CC_Licence_Link` — URL to the Creative Commons licence deed, if applicable
- [x] `licence_Copyright_Holder` — Name of the copyright holder
- [x] `licence_Copyright_Years` — Year(s) covered by the copyright
- [x] `licence_Translation_by` — Translating organisation or individual
- [ ] `licence_date_read` — Internal: date the licence file was last parsed by the pipeline

---

## STATUS_COLUMNS
Internal pipeline state. Most are excluded by default as they expose local file paths and internal bookkeeping.

- [ ] `status_download_path` — Internal: local filesystem path to the downloaded ZIP
- [ ] `status_download_date` — Internal: date the ZIP was last downloaded
- [ ] `status_unzip_path` — Internal: local path to the unzipped Paratext project folder
- [ ] `status_unzip_date` — Internal: date the project was last unzipped
- [x] `status_inferred_versification` → published as **`inferred_versification`** — Versification standard inferred for this translation (e.g. 4, 0)  
- [ ] `status_settings_xml_date` — Internal: date `Settings.xml` was last generated
- [ ] `status_extract_path` — Internal: local path to the extracted corpus `.txt` file
- [ ] `status_extract_date` — Internal: date the text was last extracted
- [ ] `status_extract_hash` — Internal: xxhash of the corpus file used for change detection
- [ ] `status_extract_hash_date` — Internal: date the hash was last computed
- [ ] `wildebeest_hash` — Internal: hash used by wildebeest normalisation tracking
- [ ] `wildebeest_hash_date` — Internal: date the wildebeest hash was last computed
- [ ] `status_last_error` — Internal: last error message from pipeline processing
