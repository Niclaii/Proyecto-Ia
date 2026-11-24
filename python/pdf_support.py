"""Lectura de importes en PDF con texto seleccionable y páginas escaneadas."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf


AMOUNT_RE = re.compile(
    r"(?<![\d/])(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?!\d)"
)
CURRENCY_RE = re.compile(r"(?i)(?:\bS\s*/|\bPEN\b|\bUSD\b|\bUS\$|[$€£])")


@dataclass(frozen=True)
class AmountCandidate:
    page_number: int
    amount: str
    raw_amount: str
    context: str
    score: int


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents).strip().lower()


def normalize_amount(value: str) -> str | None:
    token = value.strip().replace(" ", "")
    if "." in token and "," in token:
        decimal_separator = "." if token.rfind(".") > token.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        token = token.replace(grouping_separator, "").replace(decimal_separator, ".")
    elif "." in token or "," in token:
        separator = "." if "." in token else ","
        whole, fraction = token.rsplit(separator, 1)
        if len(fraction) == 3 and len(whole) <= 3:
            token = whole + fraction
        else:
            token = whole + ("." + fraction if fraction else "")
    if not re.fullmatch(r"\d+(?:\.\d{1,2})?", token):
        return None
    return token


def cue_score(context: str) -> int:
    text = normalize_text(context)
    if re.search(r"\b(gran\s+total|total\s+(?:a\s+pagar|general|final|due))\b", text):
        return 100
    if re.search(r"\b(?:importe|monto)\s+total\b", text):
        return 95
    if re.search(r"\b(?:a\s+pagar|total|amount\s+due|grand\s+total)\b", text):
        return 85
    if re.search(r"\b(?:importe|monto|amount)\b", text):
        return 70
    if re.search(r"\bsubtotal\b", text):
        return 35
    if CURRENCY_RE.search(context):
        return 20
    return 0


def extract_amount_candidates(pdf_path: Path, limit: int = 12) -> tuple[int, list[AmountCandidate]]:
    """Extrae importes candidatos del texto PDF; nunca altera el documento fuente."""
    candidates_by_value: dict[tuple[int, str], AmountCandidate] = {}
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("El PDF está protegido con contraseña y no se puede leer.")
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("El PDF no contiene páginas.")
        for page_index, page in enumerate(document):
            lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
            for line_index, line in enumerate(lines):
                line_score = cue_score(line)
                line_matches = list(AMOUNT_RE.finditer(line))
                contexts: list[tuple[str, int]] = []
                if line_matches and line_score > 0:
                    contexts.append((line, line_score))
                elif line_matches and CURRENCY_RE.search(line):
                    contexts.append((line, 20))
                elif line_score > 0:
                    # Some PDFs put the amount on the line immediately below its label.
                    for next_line in lines[line_index + 1 : line_index + 3]:
                        if cue_score(next_line) >= 35:
                            break
                        if AMOUNT_RE.search(next_line):
                            context = f"{line} {next_line}"
                            contexts.append((context, line_score))
                            break

                for context, score in contexts:
                    for match in AMOUNT_RE.finditer(context):
                        normalized = normalize_amount(match.group(0))
                        if normalized is None:
                            continue
                        candidate = AmountCandidate(
                            page_number=page_index + 1,
                            amount=normalized,
                            raw_amount=match.group(0),
                            context=context,
                            score=score + (5 if CURRENCY_RE.search(context) else 0),
                        )
                        key = (page_index + 1, normalized)
                        previous = candidates_by_value.get(key)
                        if previous is None or candidate.score > previous.score:
                            candidates_by_value[key] = candidate

    candidates = list(candidates_by_value.values())
    candidates.sort(key=lambda item: (-item.score, item.page_number))
    return page_count, candidates[:limit]


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int = 160) -> np.ndarray:
    """Renderiza una sola página como BGR para el selector de región de OpenCV."""
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("El PDF está protegido con contraseña y no se puede leer.")
        if page_index < 0 or page_index >= document.page_count:
            raise IndexError("El número de página está fuera del PDF.")
        page = document.load_page(page_index)
        scale = dpi / 72.0
        largest_dimension = max(page.rect.width, page.rect.height) * scale
        if largest_dimension > 2600:
            scale *= 2600 / largest_dimension
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
