
---
## Interview:
Proposed phase structure (for discussion)                                                                                                                                                                        

Phase 1 — Country data in the pipeline                    
Scrape country info from ebible.org/Scriptures and add it to ebible_status.csv as a new pipeline step, then expose it in metadata.parquet.

Phase 2 — ISO 639-3 enrichment + country→continent mapping
Join reference tables at corpus_to_parquet.py time (not into ebible_status.csv), adding the enriched columns to metadata.parquet.

Phase 3 — Dataloader script
Utility that loads from HuggingFace, filters, and produces output suitable for ML use.

---
Interview questions

Language family (Phase 2)
The ISO 639-3 tables you downloaded at dataloader_feature/ contain macrolanguage groupings (e.g. ara → many Arabic dialects) and per-language metadata (scope, type), but not linguistic families like Indo-European or Bantu. If you want to filter by family in that traditional sense, we'd need additional data — Glottolog is the standard open source for it.

1. When you say "language family," do you mean ISO macrolanguages (dialect clusters like Arabic), or traditional linguistic families (Indo-European, Niger-Congo, Austronesian)? Or both as separate filter axes?

Country data (Phase 1)

2. Have you looked at https://ebible.org/Scriptures/ to confirm country data is present and structured well enough to scrape? Is it one country per translation, or can a language have multiple associated countries?
Answer: Yes 
3. Should "country" be stored as an ISO 3166 code, a country name, or both?

Country → continent (Phase 2)

4. Do you have a specific source for the country→continent mapping in mind, or should we create a simple CSV ourselves?

Dataloader output (Phase 3)

5. What does the output of the dataloader look like — a HuggingFace Dataset object, a pandas DataFrame, raw files on disk, or something else?
6. For train/test/val splits: should the split be at the verse level (random rows across translations), translation level (whole translations assigned to a split), or book level? This is the most consequential design  ▎  choice — verse-level is simple but causes cross-contamination if a translation appears in both train and test.

Data placement

7. For the ISO 639-3 and continent tables — these are slowly-changing reference data, not pipeline state. My proposal is to join them in corpus_to_parquet.py rather than adding them to ebible_status.csv, keeping the
status CSV leaner. Does that work for you, or do you want everything in the status CSV for one-stop inspection?

File location

8. The dataloader.md says to create spec.md / todo.md in huggingface_feature/ — that looks like a copy-paste from the previous feature doc. Should these go in dataloader_feature/ instead?


Answer 0: I realise that much more flexibility will be required than I first envisaged. Therefore we should implement these with a system that uses a separate table or set of tables. We should keep the design of the tables simple so that users can simply add their own table in order to obtain user-defined selections.  

Answer 1: Yes, I mean traditional linguistic language families such as Indo-European or Bantu.
Answer 2: Yes, https://ebible.org/Scriptures/ contains a table whose first column is 'Territory' and then the 'Language', 'Language (English)' and 'Vernacular Title'. We should be able to find one of those three in translations.csv in order to match it up with the Territory field which contains the name of the country. 
Here's the first three rows of the table, showing the column headings.
<table border="1" padding="2"><tbody><tr><td><b>Territory</b></td><td><a href="index.php?sort=l">Language</a></td><td><a href="index.php?sort=e">Language (English)</a></td><td><a href="index.php?sort=v">Vernacular Title</a></td><td><a href="index.php?sort=t">English Title</a></td></tr><tr class="redist"><td><a href="country.php?c=AL"><img src="/flags/al.png"> Albania</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Armãneashti/Arumanisht</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Armãneashti/Arumanisht</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Biblija tu limba Rrãmãnã</a></td><td><a href="details.php?id=rup" target="_blank">Aromanian Bible</a></td></tr>
<tr class="restricted"><td><a href="country.php?c=DZ"><img src="/flags/dz.png"> Algeria</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">Arabic, Algerian Spoken</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">Arabic, Algerian Spoken</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">العهد الجديد باللهجة الجزائرية</a></td><td><a href="details.php?id=arq" target="_blank">Arabic, Algerian Spoken: Arabe Algerian NT</a></td></tr>

Answer 3. Store both the country name and the ISO 3166 code.
Answer 4. Sources of data: 
Wikipedia has a table for the list of country codes : https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes We can use those to construct our additional tables. 
ISO 3166-1 – Codes for the representation of names of countries and their subdivisions – Part 1: Country codes[2] defines codes for the names of countries, dependent territories, and special areas of geographical interest. It defines three sets of country codes:
ISO 3166-1 alpha-2 – two-letter country codes which are also used to create the ISO 3166-2 country subdivision codes and the Internet country code top-level domains.
ISO 3166-1 alpha-3 – three-letter country codes which may allow a better visual association between the codes and the country names than the 3166-1 alpha-2 codes.
ISO 3166-1 numeric 

https://unstats.un.org/unsd/methodology/m49/  The list of countries or areas contains the names of countries or areas in alphabetical order, their three-digit numerical codes used for statistical processing purposes by the Statistics Division of the United Nations Secretariat, and their three-digit alphabetical codes assigned by the International Organization for Standardization (ISO). The lists are available in these languages - English, Chinese, Russian, French, Spanish, Arabic. Having these names for the countries may be a great help for international users of the dataset. It is another reason why a separate and extendable set of tables to be used by the dataloading script will be useful.

This github gist contains a list of countries by continent: https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c#file-country-and-continent-codes-list-csv-csv

Answer 5. That's a great question and I'll need to investigate further and learn before I can answer it. One of the motivations for creating this dataset was to be able to reproduce Sami Liedes experiment as described in his blog: https://samiliedes.wordpress.com/author/samiliedes/  Briefly he trained a model with the Bible in ~50 languages, omitting Genesis, or the whole OT from 3 of them. Then the model 'knows' the content of the Bible having seen it in many languages. Can it translate Genesis into the languages where Genesis was omitted from the training data. This data is being prepared in order to be able to repeat variations of that experiment.  The output from the dataloader script should support that goal.

Answer 6. Train, Test and Val splits should be possible at all levels. This area of the design might require further thought. I imagine using a table named train.csv with columns 'translationId' 'book' 'chapters' 'verses'  where the user could list which translations and books are in the training dataset by omitting values in the chapters and verses columns. Similarly they could indicate down to the chapter level by omitting verse references. Or they could have very precise selections down to the verse level. That would allow a mix of verses across train, test and val that is controlled so that there isn't cross contamination.  Perhaps the dataloading script would provide a feature that helps the user to create those csv files. 

Answer 7. For the ISO 639-3 and continent tables let's join them in corpus_to_parquet.py as you suggest. This will fix the country and continent data for each version of the data to those at the time the data was created. It shouldn't be difficult to update those columns as required. 