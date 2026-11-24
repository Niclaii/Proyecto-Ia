"""Localiza campos etiquetados en imágenes y páginas PDF mediante OCR."""

from __future__ import annotations

import csv
import difflib
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from pdf_support import AMOUNT_RE, normalize_amount, render_pdf_page


FIELD_LABELS = {
    "total": ("total a pagar", "gran total", "total general", "importe total", "monto total", "total"),
    "monto": ("monto", "importe", "amount due", "amount", "a pagar"),
    "precio": ("precio unitario", "p unitario", "unit price", "precio", "unitario", "price"),
    "fecha": ("fecha de emision", "fecha de pago", "fecha", "date", "emision", "issued"),
}
FIELD_MATCH_WORDS = {
    "total": {"total"},
    "monto": {"monto", "importe", "amount"},
    "precio": {"precio", "unitario", "price"},
    "fecha": {"fecha", "date", "emision", "issued"},
}
DATE_RE = re.compile(
    r"(?<!\d)(?:(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})|(\d{4})[./-](\d{1,2})[./-](\d{1,2}))(?!\d)"
)
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?![A-Za-z0-9])"
)
PRICE_CUE_RE = re.compile(
    r"(?i)\b(?:precio|price|unitario|unit|importe|monto|total|cost|oferta|sale)\b"
)
CURRENCY_CUE_RE = re.compile(r"(?i)(?:\bS\s*/|\bPEN\b|\bUSD\b|US\$|[$€£])")
OCR_CORRECTIONS = {
    "torat": "total",
    "toral": "total",
    "tota1": "total",
    "runtaro": "unitario",
    "runtario": "unitario",
    "euntaro": "unitario",
    "tmpporte": "importe",
    "imp0rte": "importe",
    "feeha": "fecha",
    "feha": "fecha",
}
CURRENCY_WORDS = {"s", "si", "sol", "soles", "pen", "usd", "us", "dolar", "dolares"}
LABEL_STOP_WORDS = {"a", "de", "del", "la", "el", "y"}


@dataclass(frozen=True)
class OCRWord:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float
    line_key: tuple[int, int, int]


@dataclass(frozen=True)
class DocumentFieldCandidate:
    field: str
    page_number: int
    label: str
    value: str | None
    context: str
    confidence: float
    score: float
    bounds: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class CameraNumberCandidate:
    """A numeric-only OCR result from the camera guide area."""

    value: str
    confidence: float
    score: float
    price_cued: bool


@dataclass(frozen=True)
class CameraNumberScan:
    candidates: list[CameraNumberCandidate]
    has_alphabetic_text: bool


def _find_tesseract() -> str:
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured and Path(configured).is_file():
        return configured
    located = shutil.which("tesseract")
    if located:
        return located
    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "No encuentro Tesseract OCR. Instálalo con `winget install --id tesseract-ocr.tesseract --exact` "
        "y vuelve a abrir la aplicación."
    )


def _ocr_words(image: np.ndarray) -> list[OCRWord]:
    executable = _find_tesseract()
    height, width = image.shape[:2]
    scale = max(1.0, min(3.0, 1800.0 / max(height, width)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("No pude preparar la imagen para el OCR.")
    run = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "11", "-l", "eng", "tsv"],
        input=encoded.tobytes(),
        capture_output=True,
        timeout=90,
        check=False,
    )
    if run.returncode != 0:
        detail = run.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract no pudo leer la imagen. {detail}".strip())

    words: list[OCRWord] = []
    reader = csv.DictReader(io.StringIO(run.stdout.decode("utf-8", errors="replace")), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
            x0 = round(int(row["left"]) / scale)
            y0 = round(int(row["top"]) / scale)
            x1 = round((int(row["left"]) + int(row["width"])) / scale)
            y1 = round((int(row["top"]) + int(row["height"])) / scale)
            key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (KeyError, TypeError, ValueError):
            continue
        if confidence >= 0 and x1 > x0 and y1 > y0:
            words.append(OCRWord(text, x0, y0, x1, y1, confidence / 100.0, key))
    return words


def _pdf_words(page: pymupdf.Page) -> list[OCRWord]:
    words = []
    for item in page.get_text("words"):
        x0, y0, x1, y1, text, block_number, line_number, _word_number = item[:8]
        words.append(OCRWord(
            str(text), round(x0), round(y0), round(x1), round(y1), 1.0,
            (int(block_number), 0, int(line_number)),
        ))
    return words


def _clean_word(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", value.lower())
    return OCR_CORRECTIONS.get(cleaned, cleaned)


def _alias_words(field: str) -> set[str]:
    return {_clean_word(word) for word in FIELD_MATCH_WORDS[field]}


def _label_word_score(word: str, field: str) -> float:
    cleaned = _clean_word(word)
    if not cleaned:
        return 0.0
    aliases = _alias_words(field)
    if cleaned in aliases:
        return 1.0
    if len(cleaned) < 4:
        return 0.0
    return max(
        (difflib.SequenceMatcher(None, cleaned, alias).ratio() for alias in aliases if len(alias) >= 4),
        default=0.0,
    )


def _group_lines(words: list[OCRWord]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(word.line_key, []).append(word)
    lines = []
    for key, line_words in grouped.items():
        line_words.sort(key=lambda word: (word.x0, word.y0))
        lines.append({
            "key": key,
            "words": line_words,
            "text": " ".join(word.text for word in line_words),
            "x0": min(word.x0 for word in line_words),
            "y0": min(word.y0 for word in line_words),
            "x1": max(word.x1 for word in line_words),
            "y1": max(word.y1 for word in line_words),
            "confidence": float(np.mean([word.confidence for word in line_words])),
        })
    return sorted(lines, key=lambda line: (int(line["y0"]), int(line["x0"])))


def _find_anchor(line: dict[str, object], field: str) -> tuple[list[OCRWord], float] | None:
    words = line["words"]
    assert isinstance(words, list)
    cleaned_words = [_clean_word(word.text) for word in words]
    phrase_matches: list[tuple[list[OCRWord], float]] = []
    for alias in FIELD_LABELS[field]:
        parts = [_clean_word(part) for part in alias.split()]
        if not parts:
            continue
        phrase_score = min(1.0, 0.75 + len(" ".join(parts)) * 0.025)
        for start in range(len(words) - len(parts) + 1):
            end = start + len(parts)
            if cleaned_words[start:end] == parts:
                phrase_words = words[start:end]
                if all(word.confidence >= 0.12 for word in phrase_words):
                    phrase_matches.append((phrase_words, phrase_score))

    matched = [(word, _label_word_score(word.text, field)) for word in words]
    matched = [
        (word, score) for word, score in matched
        if score >= 0.82 and word.confidence >= 0.12
    ]
    if matched:
        anchors = [word for word, _score in matched]
        score = max(value for _word, value in matched)
        return anchors, score
    if not phrase_matches:
        return None
    anchors, score = max(phrase_matches, key=lambda match: match[1])
    return anchors, score


def _parse_dates(text: str) -> list[tuple[str, str]]:
    values = []
    for match in DATE_RE.finditer(text):
        if match.group(1) is not None:
            day, month, year = (int(match.group(i)) for i in (1, 2, 3))
        else:
            year, month, day = (int(match.group(i)) for i in (4, 5, 6))
        if year < 100:
            year += 2000
        try:
            value = date(year, month, day).isoformat()
        except ValueError:
            continue
        values.append((value, match.group(0)))
    return values


def _parse_values(text: str, field: str) -> list[tuple[str, str]]:
    if field == "fecha":
        return _parse_dates(text)
    values = []
    for match in AMOUNT_RE.finditer(text):
        normalized = normalize_amount(match.group(0))
        if normalized is not None:
            values.append((normalized, match.group(0)))
    return values


def _nearby_value_lines(
    lines: list[dict[str, object]], anchor: dict[str, object], field: str, width: int, height: int
) -> list[tuple[dict[str, object], float]]:
    results = []
    ax0, ay0, ax1, ay1 = (int(anchor[name]) for name in ("x0", "y0", "x1", "y1"))
    anchor_key = anchor["key"]
    anchor_height = max(1, ay1 - ay0)
    for line in lines:
        if line["key"] == anchor_key:
            results.append((line, 1.0))
            continue
        lx0, ly0, lx1, ly1 = (int(line[name]) for name in ("x0", "y0", "x1", "y1"))
        dy = ly0 - ay1
        x_gap = max(0, max(ax0 - lx1, lx0 - ax1))
        max_dy = max(anchor_height * 5, int(height * (0.12 if field in {"precio", "monto"} else 0.08)))
        max_dx = max(anchor_height * 5, int(width * 0.20))
        if 0 <= dy <= max_dy and x_gap <= max_dx:
            proximity = 0.8 - 0.35 * (dy / max(1, max_dy)) - 0.25 * (x_gap / max(1, max_dx))
            results.append((line, max(0.1, proximity)))
    return results


def _money_region(
    lines: list[dict[str, object]], anchor: dict[str, object], anchor_words: list[OCRWord],
    field: str, width: int, height: int,
) -> tuple[int, int, int, int]:
    ax0 = min(word.x0 for word in anchor_words)
    ay0 = min(word.y0 for word in anchor_words)
    ax1 = max(word.x1 for word in anchor_words)
    ay1 = max(word.y1 for word in anchor_words)
    label_height = max(6, ay1 - ay0)
    line_words = anchor["words"]
    assert isinstance(line_words, list)
    currency_right = max(
        (word.x1 for word in line_words if _clean_word(word.text) in CURRENCY_WORDS and word.x0 >= ax1 - 2),
        default=ax1,
    )
    def is_any_label(word: OCRWord) -> bool:
        return any(_label_word_score(word.text, other_field) >= 0.82 for other_field in FIELD_LABELS)

    right_words = [
        word for word in line_words
        if word.x0 >= ax1 - 2
        and not is_any_label(word)
        and _clean_word(word.text) not in CURRENCY_WORDS
        and word.y1 - word.y0 >= max(6, label_height * 1.20)
    ]

    below_words = []
    if not right_words and field == "precio":
        for line, _proximity in _nearby_value_lines(lines, anchor, field, width, height):
            if line["key"] == anchor["key"]:
                continue
            line_tokens = line["words"]
            assert isinstance(line_tokens, list)
            below_words.extend(
                word for word in line_tokens
                if word.x1 >= ax0 - label_height and word.x0 <= ax1 + label_height
            )

    if below_words:
        pad = max(3, round(label_height * 0.30))
        left = max(0, min(word.x0 for word in below_words) - pad)
        top = max(0, min(word.y0 for word in below_words) - pad)
        right = min(width, max(word.x1 for word in below_words) + pad)
        bottom = min(height, max(word.y1 for word in below_words) + pad)
        return left, top, right, bottom

    if right_words:
        max_word_height = max(word.y1 - word.y0 for word in right_words)
        start = max(ax1, currency_right) + max(2, round(label_height * 0.45))
        left = max(0, min(start, min(word.x0 for word in right_words)))
        top = max(0, min(word.y0 for word in right_words))
        right = min(width, max(word.x1 for word in right_words) + max(8, round(max_word_height * 1.30)))
        bottom = min(height, max(word.y1 for word in right_words) + max(5, round(max_word_height * 0.30)))
        # OCR can detect only the upper strokes of handwritten digits.
        bottom = min(height, max(bottom, ay1 + round(label_height * 1.0)))
    else:
        pad_x = max(8, round(label_height * 1.2))
        left = max(0, ax1 + max(2, round(label_height * 0.45)))
        right = min(width, max(left + 12, ax1 + pad_x * 5))
        if field == "total":
            right = min(width, max(right, ax1 + round(width * 0.22)))
        top = max(0, ay0 - max(3, round(label_height * 0.6)))
        bottom = min(height, ay1 + max(10, round(label_height * 2.5)))

    if right - left < 10:
        right = min(width, left + 10)
    if bottom - top < 8:
        bottom = min(height, top + 8)
    return left, top, right, bottom


def _find_field_candidates(
    words: list[OCRWord], field: str, page_number: int, width: int, height: int,
) -> list[DocumentFieldCandidate]:
    lines = _group_lines(words)
    candidates: list[DocumentFieldCandidate] = []
    for anchor in lines:
        anchor_match = _find_anchor(anchor, field)
        if anchor_match is None:
            continue
        anchor_words, label_score = anchor_match
        label = " ".join(word.text for word in anchor_words)
        label_confidence = float(np.mean([word.confidence for word in anchor_words]))
        if label_confidence < 0.12:
            continue
        anchor_x0 = min(word.x0 for word in anchor_words)
        anchor_x1 = max(word.x1 for word in anchor_words)
        anchor_height = max(6, max(word.y1 for word in anchor_words) - min(word.y0 for word in anchor_words))
        value_lines = _nearby_value_lines(lines, anchor, field, width, height)
        found = []
        for value_line, proximity in value_lines:
            confidence = float(value_line["confidence"])
            if confidence < 0.30:
                continue
            value_words = value_line["words"]
            assert isinstance(value_words, list)
            if value_line["key"] == anchor["key"]:
                value_words = [word for word in value_words if word.x0 >= anchor_x1 - 2]
            else:
                horizontal_slop = max(anchor_height * 5, int(width * 0.08))
                value_words = [
                    word for word in value_words
                    if anchor_x0 - horizontal_slop <= word.x0 <= anchor_x1 + horizontal_slop
                ]
            if not value_words:
                continue
            text = " ".join(word.text for word in value_words)
            values = _parse_values(text, field)
            if not values:
                continue
            for value, raw in values:
                score = label_score * 100 + proximity * 20 + confidence * 0.1
                found.append(DocumentFieldCandidate(
                    field, page_number, label, value, f"{anchor['text']} {text}".strip(),
                    confidence, score,
                ))
        if found:
            candidates.extend(found)
            continue

        if field != "fecha":
            bounds = _money_region(lines, anchor, anchor_words, field, width, height)
            candidates.append(DocumentFieldCandidate(
                field, page_number, label, None, str(anchor["text"]),
                label_confidence, label_score * 100, bounds,
            ))
        else:
            candidates.append(DocumentFieldCandidate(
                field, page_number, label, None, str(anchor["text"]),
                label_confidence, label_score * 100,
            ))

    deduplicated: dict[tuple[str, int, str, tuple[int, int, int, int] | None], DocumentFieldCandidate] = {}
    for candidate in candidates:
        key = (candidate.field, candidate.page_number, candidate.value or "", candidate.bounds)
        previous = deduplicated.get(key)
        if previous is None or candidate.score > previous.score:
            deduplicated[key] = candidate
    return sorted(deduplicated.values(), key=lambda candidate: (-candidate.score, candidate.page_number))


def search_image_fields(image: np.ndarray, target_field: str) -> list[DocumentFieldCandidate]:
    """Busca la categoría pedida y la fecha opcional para rellenar un gasto."""
    if target_field not in FIELD_LABELS:
        raise ValueError(f"Categoría de documento desconocida: {target_field}")
    words = _ocr_words(image)
    fields = [target_field] if target_field == "fecha" else [target_field, "fecha"]
    candidates = []
    for field in fields:
        candidates.extend(_find_field_candidates(words, field, 1, image.shape[1], image.shape[0]))
    return candidates


def search_pdf_fields(pdf_path: Path, target_field: str) -> tuple[int, list[DocumentFieldCandidate]]:
    """Busca el campo por texto PDF y usa OCR solo cuando el texto no basta."""
    if target_field not in FIELD_LABELS:
        raise ValueError(f"Categoría de documento desconocida: {target_field}")
    fields = [target_field] if target_field == "fecha" else [target_field, "fecha"]
    candidates: list[DocumentFieldCandidate] = []
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("El PDF está protegido con contraseña y no se puede leer.")
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("El PDF no contiene páginas.")
        for page_index, page in enumerate(document):
            width, height = int(page.rect.width), int(page.rect.height)
            page_candidates = []
            words = _pdf_words(page)
            if words:
                for field in fields:
                    page_candidates.extend(
                        _find_field_candidates(words, field, page_index + 1, width, height)
                    )
            fields_to_ocr = [
                field for field in fields
                if not any(item.field == field and item.value is not None for item in page_candidates)
            ]
            if fields_to_ocr:
                image = render_pdf_page(pdf_path, page_index)
                ocr_words = _ocr_words(image)
                for field in fields_to_ocr:
                    candidates.extend(
                        _find_field_candidates(
                            ocr_words, field, page_index + 1, image.shape[1], image.shape[0]
                        )
                    )
                # A text-layer label with no value has point-based bounds, while
                # the rendered page uses pixels. Keep the OCR result's pixel ROI.
                page_candidates = [
                    item for item in page_candidates
                    if item.field not in fields_to_ocr or item.value is not None
                ]
            candidates.extend(page_candidates)
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.page_number))
    return page_count, candidates


def extract_camera_numeric_candidates(image: np.ndarray) -> CameraNumberScan:
    """Read numeric tokens only, ranking amounts next to price/currency cues first."""
    words = _ocr_words(image)
    lines = _group_lines(words)
    has_alphabetic_text = any(
        re.search(r"[A-Za-z]{2,}", word.text) and word.confidence >= 0.35
        for word in words
    )
    price_fields = _find_field_candidates(
        words, "precio", 1, image.shape[1], image.shape[0]
    )
    labeled_prices = {
        candidate.value for candidate in price_fields if candidate.value is not None
    }

    best_by_value: dict[str, CameraNumberCandidate] = {}
    for line in lines:
        line_text = str(line["text"])
        has_price_cue = bool(PRICE_CUE_RE.search(line_text))
        has_currency = bool(CURRENCY_CUE_RE.search(line_text))
        confidence = float(line["confidence"])
        for match in NUMERIC_TOKEN_RE.finditer(line_text):
            value = normalize_amount(match.group(0))
            if value is None:
                continue
            digits = sum(char.isdigit() for char in match.group(0))
            has_decimal = bool(re.search(r"[.,]\d{1,2}$", match.group(0)))
            price_cued = has_price_cue or has_currency or value in labeled_prices
            score = (
                (35 if has_price_cue else 0)
                + (35 if has_currency else 0)
                + (45 if value in labeled_prices else 0)
                + (14 if has_decimal else 0)
                + min(digits, 8) * 1.5
                + confidence * 10
            )
            candidate = CameraNumberCandidate(value, confidence, score, price_cued)
            previous = best_by_value.get(value)
            if previous is None or candidate.score > previous.score:
                best_by_value[value] = candidate

    for value in labeled_prices:
        if value not in best_by_value:
            best_by_value[value] = CameraNumberCandidate(value, 0.0, 100.0, True)

    return CameraNumberScan(
        sorted(
            best_by_value.values(), key=lambda item: (-item.score, -item.confidence, item.value)
        ),
        has_alphabetic_text,
    )

