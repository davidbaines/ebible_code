from typing import List, Set, Union

ALL_BOOK_IDS = [
    "GEN",
    "EXO",
    "LEV",
    "NUM",
    "DEU",
    "JOS",
    "JDG",
    "RUT",
    "1SA",
    "2SA",  # 10
    "1KI",
    "2KI",
    "1CH",
    "2CH",
    "EZR",
    "NEH",
    "EST",
    "JOB",
    "PSA",
    "PRO",  # 20
    "ECC",
    "SNG",
    "ISA",
    "JER",
    "LAM",
    "EZK",
    "DAN",
    "HOS",
    "JOL",
    "AMO",  # 30
    "OBA",
    "JON",
    "MIC",
    "NAM",
    "HAB",
    "ZEP",
    "HAG",
    "ZEC",
    "MAL",
    "MAT",  # 40
    "MRK",
    "LUK",
    "JHN",
    "ACT",
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",  # 50
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAS",
    "1PE",  # 60
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
    "REV",
    "TOB",
    "JDT",
    "ESG",
    "WIS",  # 70
    "SIR",
    "BAR",
    "LJE",
    "S3Y",
    "SUS",
    "BEL",
    "1MA",
    "2MA",
    "3MA",
    "4MA",  # 80
    "1ES",
    "2ES",
    "MAN",
    "PS2",
    "ODA",
    "PSS",
    "JSA",  # actual variant text for JOS, now in LXA text
    "JDB",  # actual variant text for JDG, now in LXA text
    "TBS",  # actual variant text for TOB, now in LXA text
    "SST",  # actual variant text for SUS, now in LXA text, 90
    "DNT",  # actual variant text for DAN, now in LXA text
    "BLT",  # actual variant text for BEL, now in LXA text
    "XXA",
    "XXB",
    "XXC",
    "XXD",
    "XXE",
    "XXF",
    "XXG",
    "FRT",  # 100
    "BAK",
    "OTH",
    "3ES",  # Used previously but really should be 2ES
    "EZA",  # Used to be called 4ES, but not actually in any known project
    "5EZ",  # Used to be called 5ES, but not actually in any known project
    "6EZ",  # Used to be called 6ES, but not actually in any known project
    "INT",
    "CNC",
    "GLO",
    "TDX",  # 110
    "NDX",
    "DAG",
    "PS3",
    "2BA",
    "LBA",
    "JUB",
    "ENO",
    "1MQ",
    "2MQ",
    "3MQ",  # 120
    "REP",
    "4BA",
    "LAO",
]

NON_CANONICAL_IDS = {
    "XXA",
    "XXB",
    "XXC",
    "XXD",
    "XXE",
    "XXF",
    "XXG",
    "FRT",
    "BAK",
    "OTH",
    "INT",
    "CNC",
    "GLO",
    "TDX",
    "NDX",
}

BOOK_NUMBERS = dict((id, i + 1) for i, id in enumerate(ALL_BOOK_IDS))

FIRST_BOOK = 1
LAST_BOOK = len(ALL_BOOK_IDS)


def book_number_to_id(number: int, error_value: str = "***") -> str:
    if number < 1 or number >= len(ALL_BOOK_IDS):
        return error_value
    index = number - 1
    return ALL_BOOK_IDS[index]


def book_id_to_number(id: str) -> int:
    return BOOK_NUMBERS.get(id.upper(), 0)


def is_nt(book_num: int) -> bool:
    return book_num >= 40 and book_num < 67

def is_ot(book_num: int) -> bool:
    return book_num < 40


def is_ot_nt(book_num: int) -> bool:
    return is_ot(book_num) or is_nt(book_num)


def is_book_id_valid(book_id: str) -> bool:
    return book_id_to_number(book_id) > 0

def get_destination_file_from_book(file, add_isocode=True):
    isocode = file.parent.name[:3]
    book = None
    for BOOK_ID in ALL_BOOK_IDS:
        if BOOK_ID in file.name:
            book = BOOK_ID
    if not book:
        return None
    else:
        book_number = book_id_to_number(book)
        # Add one to NT and DC book numbers:
        if not is_ot(book_number):
            book_number += 1
        if add_isocode:
            new_filename = f"{book_number:02}{book}{isocode}.SFM"
        else:
            new_filename = f"{book_number:02}{book}.SFM"

        return file.with_name(new_filename)
