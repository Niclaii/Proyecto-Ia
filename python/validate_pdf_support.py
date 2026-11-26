"""Prueba extracción, renderizado y flujo C++ con un PDF de varias páginas."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

from expense_app import load_model, predict_image  # noqa: E402
from pdf_support import extract_amount_candidates, render_pdf_page  # noqa: E402


def create_text_pdf(path: Path) -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((55, 65), "FACTURA ELECTRONICA", fontsize=20)
    first.insert_text((55, 110), "Fecha: 20/11/2026", fontsize=12)
    first.insert_text((55, 155), "RUC: 20123456789", fontsize=12)
    first.insert_text((55, 220), "Subtotal: S/ 18,00", fontsize=14)
    first.insert_text((55, 275), "TOTAL A PAGAR", fontsize=16)
    first.insert_text((55, 305), "S/ 21,30", fontsize=16)

    second = document.new_page()
    second.insert_text((55, 70), "RESUMEN DE PAGOS", fontsize=18)
    second.insert_text((55, 125), "TOTAL: USD 88.50", fontsize=14)
    document.save(path)
    document.close()


def main() -> int:
    executable = ROOT / "build" / "Release" / "expense_reader.exe"
    if not executable.is_file():
        raise FileNotFoundError("Compila primero build/Release/expense_reader.exe.")

    with tempfile.TemporaryDirectory(prefix="pdf_validation_", dir=ROOT / "build") as temp_dir:
        temp = Path(temp_dir)
        pdf_path = temp / "comprobante_demo.pdf"
        csv_path = temp / "salida.csv"
        create_text_pdf(pdf_path)

        page_count, candidates = extract_amount_candidates(pdf_path)
        summary = [(item.page_number, item.amount) for item in candidates]
        expected = [(1, "21.30"), (2, "88.50"), (1, "18.00")]
        if page_count != 2 or summary[:3] != expected:
            raise AssertionError(f"Candidatos inesperados: páginas={page_count}, resultados={summary}")
        print(f"Extracción PDF multipágina: {summary[:3]} — OK")

        rendered = render_pdf_page(pdf_path, 0)
        if rendered.ndim != 3 or rendered.shape[0] < 800 or rendered.shape[1] < 500:
            raise AssertionError(f"Render PDF con tamaño inesperado: {rendered.shape}")
        print(f"Renderizado de página: {rendered.shape[1]}x{rendered.shape[0]} — OK")

        camera_frame = cv2.imread(str(ROOT / "ejemplos" / "pruebas_tiempo_real"
                                      / "camara_0123_negro_en_papel.png"), cv2.IMREAD_COLOR)
        if camera_frame is None:
            raise FileNotFoundError("Ejecuta validate_pipeline.py antes de esta prueba.")
        left, top, width, height = int(640 * 0.30), int(480 * 0.34), int(640 * 0.48), int(480 * 0.36)
        scan_png = temp / "pagina_escaneada.png"
        cv2.imwrite(str(scan_png), camera_frame[top : top + height, left : left + width])
        scanned_path = temp / "documento_escaneado.pdf"
        scanned = pymupdf.open()
        page = scanned.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(scan_png))
        scanned.save(scanned_path)
        scanned.close()

        scanned_pages, scanned_candidates = extract_amount_candidates(scanned_path)
        if scanned_pages != 1 or scanned_candidates:
            raise AssertionError("El PDF de imagen debería activar el flujo de selección manual.")
        scanned_image = render_pdf_page(scanned_path, 0)
        model, device = load_model(ROOT / "models" / "digits_vit.pth")
        digit_images, colors, predictions, _ = predict_image(
            scanned_image, model, device, adaptive=False
        )
        recognized = "".join(str(int(value)) for value in predictions)
        if recognized != "0123" or len(digit_images) != 4 or colors != ["negro"] * 4:
            raise AssertionError(
                f"Falló la ruta de PDF escaneado: {recognized}, colores={colors}, "
                f"segmentos={len(digit_images)}"
            )
        print("PDF escaneado -> renderizado -> dígitos 0123/negro: OK")

        run = subprocess.run(
            [str(executable), str(pdf_path), str(csv_path)],
            input="\ns\ncomida\n",
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=90,
            cwd=ROOT,
        )
        if run.returncode != 0:
            raise AssertionError(f"Lanzador devolvió {run.returncode}:\n{run.stdout}\n{run.stderr}")
        if "Importe leído del PDF: 21.30 (página 1)." not in run.stdout:
            raise AssertionError(f"El programa no propuso el importe esperado:\n{run.stdout}")
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0]["importe"] != "21.30":
            raise AssertionError(f"CSV inesperado: {rows}")
        print("Flujo C++ -> PDF -> confirmación -> CSV: 21.30 — OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
