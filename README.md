# eBible
This repo contains some of the scripts that create the data found in the [ebible_data](https://github.com/davidbaines/ebible_data) repository. The workflow relies on SILNLP's bulk_extract_corpora.py script to perform the Paratext-project-to-text file conversion. 

## Copyright Restrictions
The Bibles are collected from eBible.org either in the Public Domain, with a Creative Commons license, or with permission of the rights holder.

## Data Format
USFM files are downloaded from [eBible.org](https://ebible.org/), one zip file per Bible. We use bulk_extract_corpora.py from [SIL's NLP repo](https://github.com/sillsdev/silnlp/tree/master/silnlp/common/) to extract the verse text into the one verse per line format. 

### File Naming Convention
The SILNLP tool `bulk_extract_corpora.py` names the extracted text files with the format `<languageCode>-<project_folder_name>.txt`. The project folder names are typically the same as the `<translationId>`, which, is often the same as the languageCode, or begins with the languageCode. This results in names like `<languageCode>-<translationId>.txt` (e.g., `eng-KJV.txt` or `aai-aai.txt`).

After the SILNLP extraction, the `ebible.py` script is run a second time. This pass renames these files to remove any redundant leading `<languageCode>-` prefix if present, or simplifies names like `<languageCode>-<languageCode>.txt` to `<languageCode>.txt`.

The **final file naming convention** for files stored in the repository and referenced by `ebible_status.csv` is:
`<translationId>.txt` (e.g., `KJV.txt`, `aai.txt`, `abt-maprik.txt`)

where:
  - `<translationId>` is the identifier taken from the `translations.csv` file downloaded from ebible.org.

### Verse References
Verse references are shown inthe in the _vref.txt_ file. The line number of the verse reference is the same for all corpus files.
GEN 1:1 is on the first line of every file, GEN 1:2 is on the second line and so on.

  - \<book\> \<chapter\>:\<verse\> (e.g., 'GEN 1:1')

where:

  - \<book\> is the 3 letter book abbreviation ([per USFM 3.0](https://ubsicap.github.io/usfm/identification/books.html));
  - \<chapter\> is the numeric chapter number;
  - \<verse\> is the numeric verse number.

### Missing Verses
Blank lines in the Bible text file indicate that the verse is empty in the source Bible. This might be because it hasn't yet been translated and published.

### Verse Ranges
If a source Bible contained a verse range, with the text of several verses grouped together, then all of the verse text from the verse range will be found in the Bible text file on the line corresponding to the first verse in the verse range.  For each additional verse in the verse range, the token '&lt;range&gt;' will be found on the corresponding line of the Bible text file.  For example, if a source Bible contained Gen. 1:1-3 as a verse range, then the first 3 lines of its Bible text file will appear as follows:

    ...verse range text...
    <range>
    <range>

## Regenerating the corpus
The corpus needs to be regularly regenerated as the data on ebible.org changes over time.
Regenerating the corpus involves a multi-step process orchestrated by the ebible.py script and SILNLP's `bulk_extract_corpora.py`.

The `ebible.py` script is run twice:
1.  **Initial Pass**: Downloads `translations.csv`, downloads translation zip files, unpacks them into project structures, and prepares them for text extraction.
2.  **SILNLP Extraction**: The `bulk_extract_corpora.py` script from SILNLP is run to extract verse text from the prepared projects.
3.  **Final Pass**: `ebible.py` is run again to rename the extracted text files to the standard `<translationId>.txt` format and update the `ebible_status.csv` file with the paths to these renamed files and relevant dates.

To run it:
```
poetry run python ebible_data/ebible_status.py
```

EBIBLE_DATA_DIR should be set in the .env file and point to a folder containing a local copy of the eBIBLE_data repo. If required folders are missing then the script create a directory structure in EBIBLE_DATA_DIR as follows:

```
├── corpus
├── downloads
├── logs
├── metadata
├── private_corpus
├── private_projects
└── projects
```

### What the script does
In simple terms, the script:

- downloads a `translations.csv` file which outlines the currently available translations (in `metadata` dir)
- downloads zip files for each translation (in `downloads` dir)
- unpacks those zip files into paratext projects (in `projects` dir)
- constructs a licence file (in `metadata` dir)

```mermaid
 flowchart TD
    Translations[translations.csv] --> |if enough verses, redistributable and downloadable| Ids
    Filter[--filter arg?] --> |keep id's matching regex| Ids
    Ids[translation id's to process] --> |determine what needs downloading| DownloadList
    DownloadList[download list]
    DownloadDir[downloads dir] --> |determine what's cached| DownloadList
    DownloadList --> |download| Ebible
    Ebible[ebible.org] --> |save zip| DownloadDir
    Ids --> |generate projects for these id's| Projects[project dirs]
    DownloadDir --> |unzip| Projects
    Projects --> |bulk_extract_corpora| CorpusDir[corpus dir]
```

### Building extracts
The building of the extracts is done by `bulk_extract_corpora` from the silnlp project.
It generates one extract file for each paratext projects.
The extracts are put into the data directory `corpus` dir.

Once you have checked the `corpus` dir, you would replace the checked in corpus dir with your newly generated one.

There are smoke tests in (test_smoke.py](./tests/test_smoke.py) to help pick up common issues. #TODO-check

### Publishing to hugging face
The corpus_to_parquet.py script will convert the ebible_data/corpus files into a parquet file ready for uploading to HuggingFace as a dataset.
It will also create a parquet file from the data in the ebible_data/metadata/eBible Corpus Metadata.xlsx file. #TODO add a data filtering and loading script for HuggingFace.

### Caching of zip files
The script caches downloaded zip files to:

- speed it up
- reduce the load on ebible.org

The zip files are suffixed with a date representing the UTC date that they were downloaded,
e.g. if translation id `eng-KJV` was downloaded on April 5th 2023, the filename would be `eng-KJV--2023-04-05.zip`.

By default, the script will use the cached data for up to 14 days after it was downloaded.
This can be overridden, e.g. to set it to 30 days use `--max_zip_age_days 30`

Additionally the flag `--force_download` will ignore the cache and download everything fresh (including the `translations.csv` file).

The `--download-only` flag is useful when you want the script to just run the download logic alone, without extracting them to paratext projects.

### Filtering examples
The `--filter REGEX` reduces down the translation id's to just those that match the regex. 

This is useful when you are debugging/testing around particular translation id's.
This example picks out every translation id starting with "grc":

```
python ebible.py -f 'grc' PATH_TO_DATA_DIRECTORY

// Output
Command line filter used to reduce translation id's to ['grcbrent', 'grcbyz', 'grcf35', 'grcmt', 'grcsbl', 'grcsr', 'grctcgnt', 'grc-tisch', 'grctr']
```

This example matches just "gpu" (and not "gupk"):

```
$ python ebible.py -f 'gup$' PATH_TO_DATA_DIRECTORY

// Output
Command line filter used to reduce translation id's to ['gup']
```

This example matches translations id's starting with "gfk" or "hbo":

```
python ebible.py -f '(gfk|hbo)' PATH_TO_DATA_DIRECTORY

// Output
Command line filter used to reduce translation id's to ['gfk', 'gfkh', 'gfks', 'hbo', 'hboWLC']
```

### Built in filtering
The script automatically excludes some translations, for example:

- if they have too few verses
- they are marked as not downloadable in `translations.csv`
- they are not redistributable (this can be overridden with `--allow_non_redistributable`)
