from typing import TypedDict
from ollama import chat

import pymupdf
import numpy as np
import re
import sys
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"
DESCRIPTIONS_FILE = DATA_DIR / "table_descriptions.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"

PAGES_SOMMAIRE = 3

TITLE_SIZES = {"chapter": 40, "section": 16, "subsection": 10}
TOLERANCE_SIZE = 0.5
SECTION_NUMBER = re.compile(r'^\d{1,2}(\.\d{1,2})+\s')

LEFT_COLUMN_MAX_X = 280

DESCRIBE_PROMPT = ("Tu décris des tableaux extraits du règlement HYROX. "
    "En une seule phrase en français, dis ce que contient le tableau, "
    "avec les mots spécifiques à l'HYROX "
    "(par exemple : distance, poids, répétitions, points, pénalité, division, station). "
    "Ne recopie pas les valeurs du tableau. "
    "N'invente rien."
)

DESCRIBE_MODEL = 'ministral-3:3b'

class Chunk(TypedDict):
    chapter: str
    section: str
    page: int
    description_table: str
    text: str
    source: str # name of the source file


def describe_table(title: str, table) :
    description = chat(
        model = DESCRIBE_MODEL,
        messages = [{'role': 'system', 'content': DESCRIBE_PROMPT},
                    {'role': 'user', 'content': f"Section : {title}\nTableau :\n{table}"}]
        )
    return clean(description.message.content)

def section_chunks (chapter: str, section : str, text : str, page : int, source : str, chunk_size : int = 100, overlap : int = 20) -> list[Chunk]:
    """ 
    chunk_size  : number of words per chunk
    overlap : number of words repeated at the start of the next chunk
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if section :
        header = f"{chapter} > {section} :" 
    else :
        header = chapter
        section = chapter

    words = text.split()
    chunks: list[Chunk] = []
    start = 0

    while start < len(words) : 
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start : end]).strip()
        chunks.append({
            'chapter': chapter,
            'section': section,
            'page': page,
            'description_table': "",
            'text': f"{header}\n{chunk}",
            'source': source
        })

        if end == len(words):          # fin atteinte : pas de chunk en double
            break
        
        start += chunk_size - overlap

    return chunks

def level_size(size : int) -> bool :
    for level in TITLE_SIZES :
        if abs(size - TITLE_SIZES[level]) <= TOLERANCE_SIZE :
            return level
    return None

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or "").strip()


def box_text(box: dict) -> str:
    return clean(" ".join(span['text'] for line in box['textlines'] or [] for span in line['spans']))


def box_titles(box: dict) -> tuple[str, str]:
    chapter, section, other = "", "", ""
    for line in box['textlines'] or []:

        for span in line['spans']:
            if line['bbox'][0] < LEFT_COLUMN_MAX_X :
                level = level_size(span['size'])
            else :
                level = None
            if level == "chapter":
                chapter += " " + span['text']
            elif level is not None:
                section += " " + span['text']
            else : 
                other += " " + span['text']
    # à la fin de box_titles, avant le return
    section = clean(section)
    if section and not SECTION_NUMBER.match(section):
        other = section + " " + other
        section = ""
    return clean(chapter), clean(section), clean(other)


def table_chunk(chapter: str, section: str, box: dict, descriptions: dict[str, str], page: int, source: str) -> Chunk:
    if section :
        header = f"{chapter} > {section} :" 
    else :
        header = chapter
        section = chapter

    key = f"{page} : {' | '.join(clean(cell) for cell in box['table']['extract'][0])}"
    if key not in descriptions:
        descriptions[key] = describe_table(header, box['table']['markdown'])
    description = descriptions[key]

    return {
        'chapter': chapter,
        'section': section,
        'page': page,
        'description_table': f"{header}\n{description}",
        'text': f"{header}\n{box['table']['markdown']}",
        'source': source,
    }


def chunk_box(data: dict, source_file: str, descriptions: dict[str, str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    chapter = ""
    section = ""
    section_page = PAGES_SOMMAIRE
    text = ""

    for page in data['pages'][PAGES_SOMMAIRE:]:
        page_number = page['page_number']
        for box in page['boxes']:
            if box['boxclass'] in ('page-footer', 'picture', 'caption'):
                continue
            if box['boxclass'] not in ('text', 'table'):
                new_chapter, new_section, other = box_titles(box)
                if new_chapter or new_section:       # boîte de titre : on découpe la section en cours
                    all_chunks.extend(section_chunks(chapter=chapter, section=section, text=text,
                                                        page=section_page, source=source_file))
                    text = other
                    section_page = page_number
                    if new_chapter:                  # nouveau chapitre : on vide la section
                        chapter, section = new_chapter, ""
                    if new_section:
                        section = new_section
                    continue

            if box['boxclass'] == 'table' and box['table']:
                all_chunks.append(table_chunk(chapter, section, box, descriptions, page_number, source_file))
                continue

            b_text = box_text(box)
            if len(b_text) < 10 :
                continue

            text += " " + b_text

    all_chunks.extend(section_chunks(chapter=chapter, section=section, text=text,
                                        page=section_page, source=source_file))
    
    return all_chunks


def chunk_pdf(pdf : str) -> list[Chunk] :
    if CHUNKS_FILE.exists():                               # déjà fait : on relit le fichier
        return load_chunks()

    data = load_doc(pdf)
    descriptions = load_descriptions()
    chunks = chunk_box(data, Path(pdf).name, descriptions)
    save_descriptions(descriptions)
    save_chunks(chunks)
    return chunks

def load_doc(pdf : str) :
    path_pdf = Path(pdf)
    if not path_pdf.exists() : 
        print(f"File : {pdf} not found")
        sys.exit(1)
    
    doc = pymupdf.open(path_pdf)
    doc_json = doc.to_json(use_ocr=False)
    
    return json.loads(doc_json)


def load_chunks() -> list[Chunk]:
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

def save_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

def load_descriptions() -> dict[str, str]:
    if DESCRIPTIONS_FILE.exists():
        return json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))
    return {}

def save_descriptions(descriptions: dict[str, str]) -> None:
    DESCRIPTIONS_FILE.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__" :
    try :
        pdf = (sys.argv[1])
    except :
        print(f"Usage : {sys.argv[0]} file.pdf")
        sys.exit(1)

    chunks = chunk_pdf(pdf=pdf)

    for chunk in chunks[10:15] :
        print(f"Chunk : {chunk}\n")
