"""Revisa importes en imágenes, cámara o documentos PDF y los registra al confirmarlos."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import torch

from document_search import scan_camera_frame
from train import DigitVisionTransformer

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def _prepare_binary(image: np.ndarray, adaptive: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    color_frame = image if image.ndim == 3 else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if color_frame is not None else image
    if adaptive:
        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV) if color_frame is not None else None
        if float(np.median(gray)) < 127:
            # Value keeps bright saturated colors visible, even blue strokes whose
            # grayscale intensity is naturally low.
            foreground = hsv_frame[:, :, 2] if hsv_frame is not None else gray
            otsu_threshold, _ = cv2.threshold(
                foreground, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            threshold = max(float(np.median(foreground)) + 5, otsu_threshold - 10)
            _, binary = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)
        else:
            block_size = min(51, max(15, int(min(gray.shape) * 0.20) | 1))
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                block_size, 9,
            )
            if hsv_frame is not None:
                # Saturated ink can have nearly the same brightness as white paper.
                chromatic_ink = cv2.threshold(
                    hsv_frame[:, :, 1], 45, 255, cv2.THRESH_BINARY
                )[1]
                binary = cv2.bitwise_or(binary, chromatic_ink)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    if adaptive:
        # Quita bordes del teléfono/pantalla que cruzan la caja y conectan los dígitos.
        line_width = max(31, int(gray.shape[1] * 0.40))
        horizontal_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1)),
        )
        binary = cv2.subtract(binary, horizontal_lines)
    return gray, color_frame, binary


def prepare_digits(image: np.ndarray, adaptive: bool = False) -> tuple[list[np.ndarray], list[str]]:
    gray, color_frame, binary = _prepare_binary(image, adaptive)

    # In camera mode, close small breaks in strokes without fusing nearby digits.
    if adaptive:
        kernel_width = max(3, min(7, int(min(gray.shape) * 0.02) | 1))
        kernel_height = max(5, min(9, int(min(gray.shape) * 0.03) | 1))
    else:
        kernel_width = max(3, min(11, int(min(gray.shape) * 0.02) | 1))
        kernel_height = max(5, min(11, int(min(gray.shape) * 0.02) | 1))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(8, gray.shape[0] * gray.shape[1] // 1800)
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        x, y, width, height = box
        if box[2] * box[3] < min_area:
            continue
        if adaptive:
            # Reject screen edges and small interface icons; digits should occupy
            # a substantial part of the camera guide box.
            if height < gray.shape[0] * 0.18:
                continue
            if x <= 1 or x + width >= gray.shape[1] - 1:
                continue
        elif height < gray.shape[0] // 10:
            continue
        boxes.append(box)
    boxes.sort(key=lambda box: box[0])

    # Une partes de un mismo glifo, tanto separadas vertical como horizontalmente.
    vertical_gap_limit = max(6, int(gray.shape[0] * (0.05 if adaptive else 0.04)))
    horizontal_gap_limit = max(3, int(gray.shape[0] * (0.02 if adaptive else 0.025)))
    merged = True
    while merged:
        merged = False
        for first in range(len(boxes)):
            x1, y1, w1, h1 = boxes[first]
            for second in range(first + 1, len(boxes)):
                x2, y2, w2, h2 = boxes[second]
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                overlap_x_ratio = 0.2 if adaptive else 0.3
                overlap_y_ratio = 0.35 if adaptive else 0.5
                joins_vertically = (
                    overlap_x >= overlap_x_ratio * min(w1, w2) and gap_y <= vertical_gap_limit
                )
                joins_horizontally = (
                    overlap_y >= overlap_y_ratio * min(h1, h2) and gap_x <= horizontal_gap_limit
                )
                if joins_vertically or joins_horizontally:
                    left, top = min(x1, x2), min(y1, y2)
                    right, bottom = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
                    boxes[first] = (left, top, right - left, bottom - top)
                    boxes.pop(second)
                    merged = True
                    break
            if merged:
                break
    boxes.sort(key=lambda box: box[0])

    digits: list[np.ndarray] = []
    colors: list[str] = []
    for x, y, width, height in boxes:
        crop = (
            binary[y : y + height, x : x + width]
            if adaptive else 255 - gray[y : y + height, x : x + width]
        )
        scale = 20.0 / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        left, top = (28 - resized.shape[1]) // 2, (28 - resized.shape[0]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            shift = np.float32([[1, 0, 13.5 - center_x], [0, 1, 13.5 - center_y]])
            canvas = cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR)
        digits.append(canvas.astype(np.float32) / 255.0)
        colors.append(dominant_color_name(color_frame, binary, (x, y, width, height)))

    if not digits:
        raise ValueError("No se detectaron dígitos; prueba con tinta oscura y mejor contraste.")
    return digits, colors


def detect_decimal_separator(image: np.ndarray, adaptive: bool = True) -> int | None:
    """Return how many digits precede a clearly visible handwritten decimal dot/comma."""
    gray, _color_frame, binary = _prepare_binary(image, adaptive)
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = gray.shape[:2]
    digit_boxes = []
    small_marks = []
    min_area = max(2, image_height * image_width // 30000)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < min_area or x <= 1 or x + width >= image_width - 1:
            continue
        if height >= image_height * 0.18:
            digit_boxes.append((x, y, width, height))
        elif 2 <= height <= image_height * 0.18 and width <= image_height * 0.18:
            small_marks.append((x, y, width, height, area))
    digit_boxes.sort(key=lambda box: box[0])
    if len(digit_boxes) < 2:
        return None

    baseline = float(np.median([y + height for _x, y, _width, height in digit_boxes]))
    typical_height = float(np.median([height for _x, _y, _width, height in digit_boxes]))
    possible = []
    for x, y, width, height, area in small_marks:
        center_x = x + width / 2
        center_y = y + height / 2
        if not baseline - typical_height * 0.20 <= center_y <= baseline + typical_height * 0.12:
            continue
        for index, (left_x, _left_y, left_w, _left_h) in enumerate(digit_boxes[:-1]):
            right_x = digit_boxes[index + 1][0]
            if not left_x + left_w - 2 <= center_x <= right_x + 2:
                continue
            digits_right = len(digit_boxes) - index - 1
            if digits_right not in (1, 2):
                continue
            possible.append((area, index + 1))
            break
    if not possible:
        return None
    return max(possible, key=lambda item: item[0])[1]


def dominant_color_name(
    color_frame: np.ndarray | None,
    binary: np.ndarray,
    box: tuple[int, int, int, int],
) -> str:
    """Estima el color del trazo dentro de una caja usando los píxeles segmentados."""
    if color_frame is None:
        return "no disponible"

    x, y, width, height = box
    stroke_mask = binary[y : y + height, x : x + width] > 0
    pixels = color_frame[y : y + height, x : x + width][stroke_mask]
    if pixels.size == 0:
        return "no detectado"

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturated = hsv[hsv[:, 1] >= 45]
    if len(saturated) >= max(3, int(len(hsv) * 0.12)):
        color_names = []
        for hue in saturated[:, 0]:
            hue = int(hue)
            if hue < 8 or hue >= 170:
                color_names.append("rojo")
            elif hue < 20:
                color_names.append("naranja")
            elif hue < 35:
                color_names.append("amarillo")
            elif hue < 85:
                color_names.append("verde")
            elif hue < 100:
                color_names.append("cian")
            elif hue < 130:
                color_names.append("azul")
            elif hue < 155:
                color_names.append("morado")
            else:
                color_names.append("rosa")
        return Counter(color_names).most_common(1)[0][0]

    brightness = float(np.median(hsv[:, 2]))
    if brightness >= 190:
        return "blanco"
    if brightness <= 55:
        return "negro"
    return "gris"


def prepare_document_digits(image: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """Segmenta cifras de una zona de documento suprimiendo reglas del formulario."""
    color_frame = image if image.ndim == 3 else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if color_frame is not None else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Quita las líneas horizontales de tablas que suelen conectar cifras manuscritas.
    width = min(gray.shape[1], max(15, int(gray.shape[1] * 0.42)))
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)),
    )
    binary = cv2.subtract(binary, horizontal_lines)

    # En recibos, el borde izquierdo del recuadro suele confundirse con el primer dígito.
    # Limita la supresión vertical a la franja exterior izquierda para no borrar trazos.
    line_height = min(gray.shape[0], max(8, int(gray.shape[0] * 0.70)))
    vertical_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_height)),
    )
    left_edge = max(2, int(gray.shape[1] * 0.04))
    vertical_lines[:, left_edge:] = 0
    binary = cv2.subtract(binary, vertical_lines)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(6, gray.shape[0] * gray.shape[1] // 1800)
    min_height = max(5, int(gray.shape[0] * 0.30))
    boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width * box_height < min_area or box_height < min_height or box_width < 2:
            continue
        boxes.append((x, y, box_width, box_height))
    boxes.sort(key=lambda box: box[0])

    # Dos ceros escritos juntos pueden compartir un contorno; se dividen solo
    # los componentes inusualmente anchos, pues cada dígito suele ser más alto.
    split_boxes = []
    for x, y, box_width, box_height in boxes:
        if box_width >= box_height * 1.05 and box_width >= 9:
            cut = box_width // 2
            if cut >= 3 and box_width - cut >= 3:
                split_boxes.extend(((x, y, cut, box_height),
                                    (x + cut, y, box_width - cut, box_height)))
                continue
        split_boxes.append((x, y, box_width, box_height))
    split_boxes.sort(key=lambda box: box[0])

    digits: list[np.ndarray] = []
    colors: list[str] = []
    for x, y, box_width, box_height in split_boxes:
        crop = binary[y : y + box_height, x : x + box_width]
        scale = 20.0 / max(box_width, box_height)
        size = (max(1, round(box_width * scale)), max(1, round(box_height * scale)))
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        left, top = (28 - resized.shape[1]) // 2, (28 - resized.shape[0]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            shift = np.float32([[1, 0, 13.5 - center_x], [0, 1, 13.5 - center_y]])
            canvas = cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR)
        digits.append(canvas.astype(np.float32) / 255.0)
        colors.append(dominant_color_name(color_frame, binary, (x, y, box_width, box_height)))

    if not digits:
        raise ValueError("No se detectaron dígitos en el recorte; marca más cerca del importe.")
    return digits, colors


def capture_from_camera() -> np.ndarray | None:
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Revisa el permiso y que esté conectada.")

    captured = None
    stop_recognition = Event()
    recognition_lock = Lock()
    recognition_state: dict[str, object] = {
        "frame": None,
        "result": None,
        "error": None,
    }

    def recognize_frames() -> None:
        while not stop_recognition.is_set():
            with recognition_lock:
                frame_to_scan = recognition_state["frame"]
            if frame_to_scan is not None:
                try:
                    result = scan_camera_frame(frame_to_scan)
                except Exception as exc:
                    with recognition_lock:
                        recognition_state["error"] = str(exc)
                else:
                    with recognition_lock:
                        recognition_state["result"] = result
                        recognition_state["error"] = None
            stop_recognition.wait(0.65)

    recognition_thread = Thread(target=recognize_frames, name="camera-ocr", daemon=True)
    recognition_thread.start()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("La cámara dejó de enviar imágenes.")
            height, width = frame.shape[:2]
            roi_width, roi_height = int(width * 0.48), int(height * 0.36)
            left, top = int(width * 0.30), int(height * 0.34)
            right, bottom = left + roi_width, top + roi_height
            roi = frame[top:bottom, left:right].copy()
            with recognition_lock:
                recognition_state["frame"] = roi
                result = recognition_state["result"]
                error = recognition_state["error"]
            preview = frame.copy()
            cv2.rectangle(preview, (left, top), (right, bottom), (0, 220, 80), 2)
            cv2.putText(preview, "Muestra una fecha o precio dentro del recuadro", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)
            cv2.putText(preview, "ESPACIO: capturar   ESC: cancelar", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)
            if result is not None:
                status = result.display
                if result.confidence:
                    status += f" ({result.confidence:.0%})"
            elif error:
                status = "OCR no disponible"
            else:
                status = "Analizando..."
            cv2.putText(preview, status, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2)
            cv2.imshow("Registro de gastos - camara en vivo", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                captured = frame[top:bottom, left:right].copy()
                break
            if key == 27:
                break
    finally:
        stop_recognition.set()
        camera.release()
        recognition_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
    return captured


def save_corrected_samples(digits: list[np.ndarray], labels: str) -> Path:
    root = Path("data/custom_digits")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for index, (digit, label) in enumerate(zip(digits, labels)):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        image = np.clip(digit * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{stamp}_{index}.png"), image)
    return root


def save_expense(
    amount: str, category: str, csv_path: Path, expense_date: str | None = None
) -> Path:
    if not re.fullmatch(r"\d+(?:[.,]\d{1,2})?", amount):
        raise ValueError("Importe inválido. Usa números y, si corresponde, hasta dos decimales.")
    category = category.strip()
    if not category:
        raise ValueError("La categoría no puede estar vacía.")

    normalized_date = date.today().isoformat()
    if expense_date and expense_date.strip():
        raw_date = expense_date.strip()
        try:
            normalized_date = date.fromisoformat(raw_date).isoformat()
        except ValueError:
            for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
                try:
                    normalized_date = datetime.strptime(raw_date, date_format).date().isoformat()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Fecha inválida. Usa AAAA-MM-DD o DD/MM/AAAA.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fecha", "importe", "categoria"])
        if new_file:
            writer.writeheader()
        writer.writerow({"fecha": normalized_date, "importe": amount, "categoria": category})
    return csv_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path,
                        help="Imagen o documento PDF que deseas revisar")
    parser.add_argument("--gui", action="store_true",
                        help="Abre el menú visual de la aplicación")
    parser.add_argument("--camera", action="store_true",
                        help="Muestra cámara en vivo y captura la región enmarcada al pulsar Espacio")
    parser.add_argument("--model", type=Path, default=Path("models/digits_vit.pth"))
    parser.add_argument("--csv", type=Path, default=Path("gastos.csv"))
    args = parser.parse_args()

    if args.gui:
        if args.camera or args.image:
            parser.error("--gui no se combina con una imagen, PDF o --camera")
        from expense_gui import run_gui

        run_gui(args.model, args.csv)
        return

    if args.camera and args.image:
        parser.error("Indica una imagen/PDF o --camera, no ambas")
    if not args.camera and not args.image:
        parser.error("Indica una imagen/PDF o usa --camera")

    amount: str | None = None
    expense_date: str | None = None
    digit_images: list[np.ndarray] = []
    colors: list[str] = []
    vision_image: np.ndarray | None = None
    adaptive = args.camera

    if args.camera:
        vision_image = capture_from_camera()
        if vision_image is None:
            print("Captura cancelada.")
            return
        try:
            camera_scan = scan_camera_frame(vision_image)
        except RuntimeError as exc:
            print(f"OCR no disponible; usaré el modo de dígitos manuscritos: {exc}")
        else:
            if camera_scan.kind == "fecha" and camera_scan.value:
                expense_date = camera_scan.value
                print(f"Fecha detectada: {camera_scan.display.removeprefix('FECHA: ')} ({expense_date}).")
                price_candidates = [candidate for candidate in camera_scan.candidates if candidate.price_cued]
                if price_candidates:
                    print("Posibles importes relacionados: " + ", ".join(
                        f"[{index}] {candidate.value}" for index, candidate in enumerate(price_candidates, 1)
                    ))
                    selection = input("Elige el importe (Enter para [1]): ").strip()
                    try:
                        selected_index = int(selection) - 1 if selection else 0
                        if selected_index < 0 or selected_index >= len(price_candidates):
                            raise ValueError
                    except ValueError as exc:
                        raise ValueError("Selección numérica no válida.") from exc
                    amount = price_candidates[selected_index].value
                else:
                    amount = input("Escribe el importe de este gasto: ").strip()
            elif camera_scan.candidates:
                print("Números detectados (se ignoraron las palabras):")
                for index, candidate in enumerate(camera_scan.candidates, start=1):
                    cue = "precio probable" if candidate.price_cued else "otro número"
                    print(f"  [{index}] {candidate.value} ({cue})")
                selected_index = 0
                if len(camera_scan.candidates) > 1:
                    selection = input("Elige el precio (Enter para [1]): ").strip()
                    try:
                        selected_index = int(selection) - 1 if selection else 0
                        if selected_index < 0 or selected_index >= len(camera_scan.candidates):
                            raise ValueError
                    except ValueError as exc:
                        raise ValueError("Selección numérica no válida.") from exc
                amount = camera_scan.candidates[selected_index].value
            elif camera_scan.has_alphabetic_text:
                use_handwriting = input(
                    "Solo detecté palabras, sin números impresos. ¿Intento leer números manuscritos? (s/n): "
                ).strip().lower()
                if use_handwriting not in {"s", "si", "sí"}:
                    print("No se propuso ningún importe.")
                    return
    elif args.image.suffix.lower() == ".pdf":
        from pdf_support import extract_amount_candidates

        if not args.image.is_file():
            raise FileNotFoundError(f"No existe el PDF: {args.image}")
        page_count, candidates = extract_amount_candidates(args.image)
        if candidates:
            print(f"PDF de {page_count} página(s): importes encontrados en el texto.")
            for index, candidate in enumerate(candidates, start=1):
                context = " ".join(candidate.context.split())
                if len(context) > 110:
                    context = context[:107] + "..."
                print(f"  [{index}] página {candidate.page_number}: {candidate.amount} | {context}")
            selection = input("Elige el importe a revisar (Enter para [1]): ").strip()
            try:
                selected_index = int(selection) - 1 if selection else 0
                if selected_index < 0 or selected_index >= len(candidates):
                    raise ValueError
            except ValueError as exc:
                raise ValueError("Selección de importe no válida.") from exc
            chosen = candidates[selected_index]
            amount = chosen.amount
            print(f"Importe leído del PDF: {amount} (página {chosen.page_number}).")
        else:
            print(f"PDF de {page_count} página(s), sin importe legible en su texto.")
            vision_image = select_pdf_digit_region(args.image, page_count)
            adaptive = False
    else:
        vision_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if vision_image is None:
            raise ValueError(f"No se pudo abrir la imagen: {args.image}")

    confidences: np.ndarray | None = None
    if vision_image is not None and amount is None:
        model, device = load_model(args.model)
        digit_images, colors, predictions, confidences = predict_image(
            vision_image, model, device, adaptive
        )
        amount = "".join(str(int(digit)) for digit in predictions)
        decimal_index = detect_decimal_separator(vision_image, adaptive=adaptive)
        if (
            decimal_index is not None
            and 0 < decimal_index < len(amount)
            and len(amount) - decimal_index <= 2
        ):
            amount = amount[:decimal_index] + "." + amount[decimal_index:]
            print(f"Separador decimal detectado después del dígito {decimal_index}.")
        print(f"Segmentos de dígito detectados: {len(digit_images)}")
        print(f"Dispositivo: {device}")
        print("Dígitos (izquierda a derecha):")
        for digit, color, confidence in zip(predictions, colors, confidences):
            print(f"  {digit}  color {color}  confianza {confidence:.1%}")
        if float(confidences.min()) < 0.55:
            print("Revisión necesaria: al menos un dígito tiene confianza baja.")

    if amount is None:
        raise RuntimeError("No se pudo obtener un importe para revisar.")
    print(f"Importe propuesto: {amount}")

    corrected = input("¿El importe es correcto? (s/n): ").strip().lower() in {"n", "no"}
    if corrected:
        amount = input("Escribe el importe corregido (punto o coma decimal): ").strip()
    if not re.fullmatch(r"\d+(?:[.,]\d{1,2})?", amount):
        raise ValueError("Importe inválido. No se guardó el gasto.")
    labels = amount.replace(".", "").replace(",", "")
    if corrected and digit_images and len(labels) == len(digit_images) and labels.isdigit():
        if input("¿Guardar estos dígitos recortados para personalizar el modelo? (s/n): ").strip().lower() in {"s", "si", "sí"}:
            sample_dir = save_corrected_samples(digit_images, labels)
            print(f"Ejemplos personales guardados en {sample_dir.resolve()}")

    category = input("Categoría (comida/transporte/hogar/otros): ").strip()
    saved_path = save_expense(amount, category, args.csv, expense_date)
    print(f"Gasto guardado en {saved_path}")


def load_model(model_path: Path) -> tuple[DigitVisionTransformer, torch.device]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No existe {model_path}. Entrena el modelo con python python/train.py primero."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitVisionTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def predict_image(
    image: np.ndarray,
    model: DigitVisionTransformer,
    device: torch.device,
    adaptive: bool,
    document: bool = False,
) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]:
    digit_images, colors = (
        prepare_document_digits(image) if document else prepare_digits(image, adaptive=adaptive)
    )
    batch = torch.from_numpy(np.stack(digit_images)[:, None, :, :]).to(device)
    with torch.no_grad():
        probabilities = model(batch).softmax(dim=1).cpu().numpy()
    return digit_images, colors, probabilities.argmax(axis=1), probabilities.max(axis=1)


def select_pdf_digit_region(pdf_path: Path, page_count: int) -> np.ndarray:
    """Muestra una página escaneada para que la persona marque el importe."""
    from pdf_support import render_pdf_page

    raw_page = input(f"Elige la página que contiene el importe (1-{page_count}, Enter para 1): ").strip()
    try:
        page_number = int(raw_page) if raw_page else 1
    except ValueError as exc:
        raise ValueError("Número de página no válido.") from exc
    if page_number < 1 or page_number > page_count:
        raise ValueError("Número de página fuera del documento.")

    page = render_pdf_page(pdf_path, page_number - 1)
    scale = min(1.0, 1100 / page.shape[1], 760 / page.shape[0])
    display = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else page
    window = "PDF escaneado - selecciona solo los digitos del importe"
    print("Marca con el mouse la zona de los dígitos; pulsa Enter para confirmar o Esc para cancelar.")
    x, y, width, height = cv2.selectROI(window, display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if width == 0 or height == 0:
        raise ValueError("No se seleccionó una región del PDF.")

    scale_x = page.shape[1] / display.shape[1]
    scale_y = page.shape[0] / display.shape[0]
    left = max(0, round(x * scale_x))
    top = max(0, round(y * scale_y))
    right = min(page.shape[1], round((x + width) * scale_x))
    bottom = min(page.shape[0], round((y + height) * scale_y))
    return page[top:bottom, left:right].copy()


if __name__ == "__main__":
    main()

