"""Menú gráfico para revisar importes y volver al menú después de cada acción."""

from __future__ import annotations

import base64
import csv
import re
import tkinter as tk
from datetime import date
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from expense_app import (
    capture_from_camera,
    detect_decimal_separator,
    load_model,
    predict_image,
    save_expense,
)
from document_search import (
    CameraNumberCandidate,
    DocumentFieldCandidate,
    scan_camera_frame,
    search_image_fields,
    search_pdf_fields,
)
from pdf_support import AmountCandidate, extract_amount_candidates, render_pdf_page


BACKGROUND = "#f2f5f9"
NAVY = "#14253d"
BLUE = "#2365d1"
GREEN = "#168c58"
TEXT = "#243247"
MUTED = "#64748b"
WHITE = "#ffffff"


class CancelAction(Exception):
    """La persona canceló un selector; no se considera un error."""


class ExpenseReaderGUI:
    def __init__(self, root: tk.Tk, model_path: Path, csv_path: Path) -> None:
        self.root = root
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.device = None
        self.busy = False
        self.status = tk.StringVar(value="Listo para revisar un gasto.")

        self.root.title("Registro de gastos")
        self.root.geometry("900x660")
        self.root.minsize(820, 600)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._confirm_exit)
        self._configure_styles()
        self._build_menu()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Section.TLabel", background=BACKGROUND, foreground=TEXT,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 9))

    def _build_menu(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=24)
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=NAVY, padx=26, pady=22)
        header.pack(fill="x")
        tk.Label(header, text="Registro de gastos", bg=NAVY, fg=WHITE,
                 font=("Segoe UI", 24, "bold")).pack(anchor="w")
        tk.Label(header, text="Busca datos en documentos o consulta tus gastos", bg=NAVY, fg="#d4deeb",
                 font=("Segoe UI", 11)).pack(anchor="w", pady=(5, 0))

        ttk.Label(shell, text="Menú principal", style="Section.TLabel").pack(anchor="w", pady=(22, 12))
        cards = tk.Frame(shell, bg=BACKGROUND)
        cards.pack(fill="both", expand=True)
        for index in range(2):
            cards.grid_columnconfigure(index, weight=1, uniform="cards")
            cards.grid_rowconfigure(index, weight=1, uniform="cards")

        options = [
            ("Revisar documento PDF", "Busca total, importe, precio o fecha dentro del documento.",
             "Elegir PDF", self.review_pdf, BLUE),
            ("Abrir imagen", "Busca total, importe, precio o fecha en una boleta.",
             "Elegir imagen", self.review_image, GREEN),
            ("Usar cámara", "Detecta en vivo si muestras una fecha, un precio u otro número.",
             "Iniciar cámara", self.review_camera, "#7c4ac7"),
            ("Ver gastos guardados", "Consulta los importes y categorías registrados en el CSV.",
             "Abrir historial", self.show_history, "#c16a20"),
        ]
        for index, (title, description, button_text, action, accent) in enumerate(options):
            card = tk.Frame(cards, bg=WHITE, highlightbackground="#dce3ec", highlightthickness=1)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=7, pady=7)
            tk.Frame(card, bg=accent, height=5).pack(fill="x")
            content = tk.Frame(card, bg=WHITE, padx=18, pady=14)
            content.pack(fill="both", expand=True)
            tk.Label(content, text=title, bg=WHITE, fg=TEXT,
                     font=("Segoe UI", 14, "bold"), anchor="w").pack(fill="x")
            tk.Label(content, text=description, bg=WHITE, fg=MUTED,
                     font=("Segoe UI", 10), anchor="nw", justify="left",
                     wraplength=340, height=2).pack(fill="x", pady=(7, 12))
            tk.Button(content, text=button_text, command=lambda fn=action: self._run_action(fn),
                      bg=accent, fg=WHITE, activebackground=accent, activeforeground=WHITE,
                      relief="flat", bd=0, padx=14, pady=8,
                      font=("Segoe UI", 10, "bold"), cursor="hand2").pack(anchor="w")

        footer = tk.Frame(shell, bg=BACKGROUND)
        footer.pack(fill="x", pady=(12, 0))
        tk.Label(footer, textvariable=self.status, bg=BACKGROUND, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Button(footer, text="Salir", command=self._confirm_exit,
                  bg="#e7ebf0", fg=TEXT, activebackground="#d8dee7", relief="flat",
                  padx=18, pady=8, font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="right")

    def _run_action(self, action) -> None:
        if self.busy:
            return
        self.busy = True
        self.root.configure(cursor="watch")
        self.status.set("Procesando…")
        self.root.update_idletasks()
        try:
            action()
        except CancelAction:
            self.status.set("Acción cancelada. Puedes elegir otra opción.")
        except Exception as exc:
            self.status.set("La acción terminó con un error.")
            messagebox.showerror("No se pudo completar", str(exc), parent=self.root)
        else:
            if self.status.get() == "Procesando…":
                self.status.set("Listo. El menú principal sigue disponible.")
        finally:
            self.busy = False
            self.root.configure(cursor="")

    def _load_model(self):
        if self.model is None:
            self.model, self.device = load_model(self.model_path)
        return self.model, self.device

    def _choose_document_field(self) -> str:
        options = {
            "Total / total a pagar": "total",
            "Monto o importe": "monto",
            "Precio unitario": "precio",
            "Fecha": "fecha",
            "Dígitos sueltos": "digits",
        }
        dialog = tk.Toplevel(self.root)
        dialog.title("Qué dato buscar")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=BACKGROUND)
        tk.Label(dialog, text="Elige qué campo localizar", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            dialog,
            text="Buscaré la etiqueta elegida y revisaré solo el valor cercano. Podrás corregirlo.",
            bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 10),
        ).pack(anchor="w", padx=18, pady=(0, 12))
        field_var = tk.StringVar(value=next(iter(options)))
        selector = ttk.Combobox(
            dialog, textvariable=field_var, values=tuple(options), state="readonly", width=34
        )
        selector.pack(fill="x", padx=18)
        result: list[str] = []

        def accept(_event=None) -> None:
            value = field_var.get()
            if value in options:
                result.append(options[value])
                dialog.destroy()

        buttons = tk.Frame(dialog, bg=BACKGROUND)
        buttons.pack(fill="x", padx=18, pady=16)
        tk.Button(buttons, text="Buscar", command=accept, bg=GREEN, fg=WHITE,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(buttons, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        selector.bind("<Return>", accept)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        selector.focus_set()
        self.root.wait_window(dialog)
        if not result:
            raise CancelAction
        return result[0]

    def _choose_document_candidate(
        self, candidates: list[DocumentFieldCandidate]
    ) -> DocumentFieldCandidate:
        if len(candidates) == 1:
            return candidates[0]
        dialog = tk.Toplevel(self.root)
        dialog.title("Elige el campo detectado")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("660x390")
        dialog.configure(bg=BACKGROUND)
        result: list[DocumentFieldCandidate] = []
        tk.Label(dialog, text="Campos encontrados", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(dialog, text="Selecciona el valor que quieras revisar.", bg=BACKGROUND,
                 fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(0, 12))
        listing = tk.Listbox(dialog, font=("Segoe UI", 10), height=10, activestyle="dotbox")
        listing.pack(fill="both", expand=True, padx=18)
        for candidate in candidates:
            value = candidate.value if candidate.value is not None else "leer cifra junto a etiqueta"
            listing.insert(
                "end",
                f"Pág. {candidate.page_number} · {candidate.label}: {value} · "
                f"{candidate.confidence:.0%} OCR",
            )
        listing.selection_set(0)

        def choose(_event=None) -> None:
            selection = listing.curselection()
            if selection:
                result.append(candidates[selection[0]])
                dialog.destroy()

        actions = tk.Frame(dialog, bg=BACKGROUND)
        actions.pack(fill="x", padx=18, pady=14)
        tk.Button(actions, text="Usar selección", command=choose, bg=BLUE, fg=WHITE,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(actions, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        listing.bind("<Double-Button-1>", choose)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.root.wait_window(dialog)
        if not result:
            raise CancelAction
        return result[0]

    @staticmethod
    def _best_scanned_date(candidates: list[DocumentFieldCandidate]) -> str | None:
        dates = [item for item in candidates if item.field == "fecha" and item.value]
        dates.sort(key=lambda item: (-item.score, item.page_number))
        return dates[0].value if dates else None

    def review_image(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona una imagen con dígitos",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos los archivos", "*.*")],
        )
        if not path:
            raise CancelAction
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo abrir esa imagen.")
        field = self._choose_document_field()
        if field == "digits":
            self._review_vision(image, Path(path).name, adaptive=False)
            return

        candidates = search_image_fields(image, field)
        matches = [candidate for candidate in candidates if candidate.field == field]
        if not matches:
            if field == "fecha":
                self._show_review(
                    "", Path(path).name, [],
                    "No encontré una etiqueta FECHA legible. Escribe la fecha del documento y el importe.",
                    date_required=True, preview_image=image,
                )
                return
            if not messagebox.askyesno(
                "No encontré la etiqueta",
                "No encontré el campo elegido. ¿Quieres marcar manualmente la zona del importe?",
                parent=self.root,
            ):
                return
            region = self._select_region(image, f"Imagen: {Path(path).name}")
            expense_date = self._best_scanned_date(candidates)
            self._review_vision(
                region, Path(path).name, adaptive=False, document=True, expense_date=expense_date
            )
            return

        candidate = self._choose_document_candidate(matches)
        expense_date = candidate.value if field == "fecha" else self._best_scanned_date(candidates)
        if candidate.bounds:
            x0, y0, x1, y1 = candidate.bounds
            region = image[y0:y1, x0:x1].copy()
            self._review_vision(
                region, Path(path).name, adaptive=False, document=True, expense_date=expense_date
            )
            return
        if candidate.value is None:
            if field == "fecha":
                self._show_review(
                    "", Path(path).name, [],
                    f"Encontré «{candidate.label}», pero no pude leer la fecha. Escríbela junto con el importe.",
                    date_required=True, preview_image=image,
                )
                return
            messagebox.showwarning(
                "No pude leer el campo",
                f"Encontré la etiqueta «{candidate.label}», pero no pude leer su valor. Prueba con mejor enfoque.",
                parent=self.root,
            )
            return

        if field == "fecha":
            detail = f"Fecha localizada junto a la etiqueta «{candidate.label}»: {candidate.value}. Escribe el importe para registrar el gasto."
            amount = ""
        else:
            detail = f"Valor localizado junto a «{candidate.label}»: {candidate.value}. Revisa que corresponda al campo elegido."
            amount = candidate.value
        self._show_review(
            amount, Path(path).name, [], detail, expense_date=expense_date
        )

    def review_camera(self) -> None:
        image = capture_from_camera()
        if image is None:
            raise CancelAction
        try:
            scan = scan_camera_frame(image)
        except RuntimeError as exc:
            messagebox.showwarning(
                "OCR no disponible",
                f"No pude leer la etiqueta con OCR. Intentaré el modo de dígitos manuscritos.\n\n{exc}",
                parent=self.root,
            )
            self._review_vision(image, "Captura de cámara", adaptive=True)
            return

        if scan.kind == "fecha" and scan.value:
            price_candidates = [candidate for candidate in scan.candidates if candidate.price_cued]
            amount = self._choose_camera_number(price_candidates).value if price_candidates else ""
            detail = f"Fecha detectada: {scan.display.removeprefix('FECHA: ')} ({scan.value})."
            detail += (
                f" Precio sugerido: {amount}. Confírmalo antes de guardar."
                if amount else " Escribe el importe relacionado para guardar el gasto."
            )
            self._show_review(
                amount, "Captura de cámara", [], detail,
                expense_date=scan.value, preview_image=image,
            )
            return

        if scan.kind == "hora" and scan.value:
            messagebox.showinfo(
                "Hora detectada",
                f"Reconocí la hora {scan.display.removeprefix('HORA: ')}.\n\n"
                "La aplicación registra fecha e importe, así que no creó un gasto con la hora como si fuera un precio.",
                parent=self.root,
            )
            return

        if scan.candidates:
            candidate = self._choose_camera_number(scan.candidates)
            if candidate.price_cued:
                detail = "Precio detectado cerca de una etiqueta o símbolo de moneda. Solo se conservó el número; revisa el recorte."
            else:
                detail = "OCR conservó solo las cifras de la captura. No vi una etiqueta de precio clara; verifica que sea el precio."
            self._show_review(
                candidate.value, "Captura de cámara", [], detail, preview_image=image
            )
            return

        if scan.has_alphabetic_text and not messagebox.askyesno(
            "No encontré un precio numérico",
            "La cámara reconoció palabras, pero no encontró cifras impresas. ¿Quieres intentar leer la captura como números manuscritos?",
            parent=self.root,
        ):
            return
        self._review_vision(image, "Captura de cámara", adaptive=True)

    def _choose_camera_number(
        self, candidates: list[CameraNumberCandidate]
    ) -> CameraNumberCandidate:
        if len(candidates) == 1:
            return candidates[0]
        dialog = tk.Toplevel(self.root)
        dialog.title("Elige el número del precio")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("430x330")
        dialog.configure(bg=BACKGROUND)
        result: list[CameraNumberCandidate] = []
        tk.Label(dialog, text="Números detectados", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(dialog, text="Las palabras se ignoraron. Selecciona el precio correcto.",
                 bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(0, 12))
        listing = tk.Listbox(dialog, font=("Segoe UI", 12), height=8, activestyle="dotbox")
        listing.pack(fill="both", expand=True, padx=18)
        for candidate in candidates:
            description = " · precio probable" if candidate.price_cued else " · otro número"
            listing.insert("end", f"{candidate.value}{description}")
        listing.selection_set(0)

        def choose(_event=None) -> None:
            selection = listing.curselection()
            if selection:
                result.append(candidates[selection[0]])
                dialog.destroy()

        actions = tk.Frame(dialog, bg=BACKGROUND)
        actions.pack(fill="x", padx=18, pady=14)
        tk.Button(actions, text="Usar número", command=choose, bg=GREEN, fg=WHITE,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(actions, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        listing.bind("<Double-Button-1>", choose)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.root.wait_window(dialog)
        if not result:
            raise CancelAction
        return result[0]

    def review_pdf(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona un documento PDF",
            filetypes=[("Documentos PDF", "*.pdf")],
        )
        if not path:
            raise CancelAction

        pdf_path = Path(path)
        field = self._choose_document_field()
        if field == "digits":
            page_count, amount_candidates = extract_amount_candidates(pdf_path)
            if amount_candidates:
                chosen = self._choose_pdf_candidate(amount_candidates)
                detail = f"Página {chosen.page_number}: {chosen.context}"
                self._show_review(chosen.amount, pdf_path.name, [], detail)
                return
            self._review_pdf_region(pdf_path, page_count)
            return

        page_count, candidates = search_pdf_fields(pdf_path, field)
        matches = [candidate for candidate in candidates if candidate.field == field]
        expense_date = self._best_scanned_date(candidates)
        if not matches:
            if field == "fecha":
                first_page = render_pdf_page(pdf_path, 0)
                self._show_review(
                    "", pdf_path.name, [],
                    "No encontré una etiqueta FECHA legible. Escribe la fecha del documento y el importe.",
                    date_required=True, preview_image=first_page,
                )
                return
            if not messagebox.askyesno(
                "No encontré la etiqueta",
                "No encontré el campo elegido. ¿Quieres marcar manualmente la zona del importe?",
                parent=self.root,
            ):
                return
            self._review_pdf_region(pdf_path, page_count, expense_date)
            return

        candidate = self._choose_document_candidate(matches)
        if field == "fecha":
            expense_date = candidate.value
        if candidate.bounds:
            page = render_pdf_page(pdf_path, candidate.page_number - 1)
            x0, y0, x1, y1 = candidate.bounds
            region = page[y0:y1, x0:x1].copy()
            self._review_vision(
                region, f"{pdf_path.name} · página {candidate.page_number}",
                adaptive=False, document=True, expense_date=expense_date,
            )
            return
        if candidate.value is None:
            if field == "fecha":
                page = render_pdf_page(pdf_path, candidate.page_number - 1)
                self._show_review(
                    "", pdf_path.name, [],
                    f"Encontré «{candidate.label}» en la página {candidate.page_number}, pero no pude leer la fecha. Escríbela junto con el importe.",
                    date_required=True, preview_image=page,
                )
                return
            messagebox.showwarning(
                "No pude leer el campo",
                f"Encontré la etiqueta «{candidate.label}», pero no pude leer su valor.",
                parent=self.root,
            )
            return
        if field == "fecha":
            amount = ""
            detail = f"Fecha localizada en página {candidate.page_number}, junto a «{candidate.label}»: {candidate.value}. Escribe el importe para registrar el gasto."
        else:
            amount = candidate.value
            detail = f"Página {candidate.page_number}. Valor junto a «{candidate.label}»: {candidate.value}. Revisa que corresponda al campo elegido."
        self._show_review(amount, pdf_path.name, [], detail, expense_date=expense_date)

    def _review_pdf_region(
        self, pdf_path: Path, page_count: int, expense_date: str | None = None
    ) -> None:
        messagebox.showinfo(
            "Selección manual",
            "Marca todos los dígitos del importe para leerlos sin usar otros números del PDF.",
            parent=self.root,
        )
        page_number = simpledialog.askinteger(
            "Elegir página", f"¿Qué página contiene el importe? (1 a {page_count})",
            parent=self.root, minvalue=1, maxvalue=page_count, initialvalue=1,
        )
        if page_number is None:
            raise CancelAction
        page = render_pdf_page(pdf_path, page_number - 1)
        region = self._select_region(page, f"PDF: página {page_number}")
        self._review_vision(
            region, f"{pdf_path.name} · página {page_number}", adaptive=False,
            document=True, expense_date=expense_date,
        )

    def _choose_pdf_candidate(self, candidates: list[AmountCandidate]) -> AmountCandidate:
        if len(candidates) == 1:
            return candidates[0]
        dialog = tk.Toplevel(self.root)
        dialog.title("Elige el importe del documento")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("660x390")
        dialog.configure(bg=BACKGROUND)
        result: list[AmountCandidate] = []

        tk.Label(dialog, text="Importes encontrados", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(dialog, text="Selecciona cuál quieres revisar.", bg=BACKGROUND, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(0, 12))
        listing = tk.Listbox(dialog, font=("Segoe UI", 10), height=11, activestyle="dotbox")
        listing.pack(fill="both", expand=True, padx=18)
        for candidate in candidates:
            context = " ".join(candidate.context.split())
            if len(context) > 90:
                context = context[:87] + "..."
            listing.insert("end", f"Pág. {candidate.page_number}   {candidate.amount}   |   {context}")
        listing.selection_set(0)

        actions = tk.Frame(dialog, bg=BACKGROUND)
        actions.pack(fill="x", padx=18, pady=14)

        def choose(_event=None) -> None:
            selection = listing.curselection()
            if selection:
                result.append(candidates[selection[0]])
                dialog.destroy()

        tk.Button(actions, text="Usar importe", command=choose, bg=BLUE, fg=WHITE,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(actions, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        listing.bind("<Double-Button-1>", choose)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.root.wait_window(dialog)
        if not result:
            raise CancelAction
        return result[0]

    def _select_region(self, image: np.ndarray, label: str) -> np.ndarray:
        scale = min(1.0, 980 / image.shape[1], 650 / image.shape[0])
        display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        if not ok:
            raise RuntimeError("No se pudo mostrar la página del PDF.")
        photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="PNG")

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Seleccionar importe — {label}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=BACKGROUND)
        tk.Label(dialog, text="Marca todos los dígitos del importe; excluye otras filas y etiquetas",
                 bg=BACKGROUND,
                 fg=TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 5))
        canvas = tk.Canvas(dialog, width=display.shape[1], height=display.shape[0],
                           highlightthickness=0, cursor="crosshair")
        canvas.pack(padx=12, pady=6)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        dialog._page_photo = photo
        selection: dict[str, object] = {"start": None, "rect": None, "coords": None}

        def start_drag(event) -> None:
            selection["start"] = (event.x, event.y)
            selection["coords"] = None
            if selection["rect"] is not None:
                canvas.delete(selection["rect"])

        def drag(event) -> None:
            start = selection["start"]
            if start is None:
                return
            x0, y0 = start
            if selection["rect"] is not None:
                canvas.delete(selection["rect"])
            selection["rect"] = canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#13a76b", width=3
            )
            selection["coords"] = (x0, y0, event.x, event.y)

        canvas.bind("<ButtonPress-1>", start_drag)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", drag)

        result: list[np.ndarray] = []

        def accept() -> None:
            coords = selection["coords"]
            if coords is None:
                messagebox.showwarning("Falta seleccionar", "Marca la zona numérica con el mouse.", parent=dialog)
                return
            x0, y0, x1, y1 = coords
            sx = image.shape[1] / display.shape[1]
            sy = image.shape[0] / display.shape[0]
            left, right = sorted((round(x0 * sx), round(x1 * sx)))
            top, bottom = sorted((round(y0 * sy), round(y1 * sy)))
            left, top = max(0, left), max(0, top)
            right, bottom = min(image.shape[1], right), min(image.shape[0], bottom)
            if right <= left or bottom <= top:
                messagebox.showwarning("Selección vacía", "Marca una zona más grande.", parent=dialog)
                return
            result.append(image[top:bottom, left:right].copy())
            dialog.destroy()

        buttons = tk.Frame(dialog, bg=BACKGROUND)
        buttons.pack(fill="x", padx=12, pady=(4, 12))
        tk.Button(buttons, text="Usar selección", command=accept, bg=GREEN, fg=WHITE,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(buttons, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=14, pady=8, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.root.wait_window(dialog)
        if not result:
            raise CancelAction
        return result[0]

    def _review_vision(
        self, image: np.ndarray, source: str, adaptive: bool, document: bool = False,
        expense_date: str | None = None,
    ) -> None:
        model, device = self._load_model()
        decimal_detected = False
        decimal_index: int | None = None
        try:
            digits, colors, predictions, confidences = predict_image(
                image, model, device, adaptive, document=document
            )
        except ValueError:
            if not document:
                raise
            amount = ""
            rows = []
            detail = "No pude separar los dígitos de esta zona. Escribe el total completo antes de guardarlo."
        else:
            amount = "".join(
                "?" if document and float(confidence) < 0.55 else str(int(digit))
                for digit, confidence in zip(predictions, confidences)
            )
            decimal_index = detect_decimal_separator(image, adaptive=adaptive)
            decimal_detected = (
                decimal_index is not None
                and amount.isdigit()
                and 0 < decimal_index < len(amount)
                and len(amount) - decimal_index <= 2
            )
            if decimal_detected:
                amount = amount[:decimal_index] + "." + amount[decimal_index:]
            rows = [
                (str(int(digit)), color, f"{float(confidence):.1%}")
                for digit, color, confidence in zip(predictions, colors, confidences)
            ]
            detail = f"El modelo detectó {len(digits)} dígito(s)."
            if float(confidences.min()) < 0.55:
                detail += " Hay una confianza baja; revisa el importe antes de guardarlo."
                if document:
                    detail += " El signo '?' marca los dígitos dudosos."
            if decimal_detected:
                detail += f" Detecté un separador decimal después del dígito {decimal_index}."
        if document:
            detail += " Es una propuesta: compara cada dígito con el recorte antes de guardarlo."
            if not decimal_detected:
                detail += " Agrega la coma o el punto decimal si corresponde."
        self._show_review(
            amount, source, rows, detail, expense_date=expense_date, preview_image=image
        )

    def _show_review(
        self, amount: str, source: str, digit_rows: list[tuple[str, str, str]],
        detail: str, expense_date: str | None = None, preview_image: np.ndarray | None = None,
        date_required: bool = False,
    ) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Revisar gasto")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=BACKGROUND)
        tk.Label(dialog, text="Confirma el importe", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=22, pady=(18, 3))
        tk.Label(dialog, text=source, bg=BACKGROUND, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=22, pady=(0, 8))
        if preview_image is not None and preview_image.size:
            preview_scale = min(
                3.0, 460 / preview_image.shape[1], 140 / preview_image.shape[0]
            )
            preview = cv2.resize(
                preview_image, None, fx=preview_scale, fy=preview_scale,
                interpolation=cv2.INTER_CUBIC if preview_scale > 1 else cv2.INTER_AREA,
            )
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB) if preview.ndim == 3 else preview
            ok, encoded = cv2.imencode(".png", preview_rgb)
            if ok:
                preview_photo = tk.PhotoImage(
                    data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="PNG"
                )
                preview_label = tk.Label(dialog, image=preview_photo, bg=WHITE)
                preview_label.image = preview_photo
                preview_label.pack(anchor="w", padx=22, pady=(0, 10))
        tk.Label(dialog, text=detail, bg=BACKGROUND, fg=MUTED,
                 font=("Segoe UI", 10), wraplength=490, justify="left").pack(anchor="w", padx=22, pady=(0, 12))

        if digit_rows:
            table = ttk.Treeview(dialog, columns=("digit", "color", "confidence"), show="headings", height=min(5, len(digit_rows)))
            table.heading("digit", text="Dígito")
            table.heading("color", text="Color")
            table.heading("confidence", text="Confianza")
            table.column("digit", width=85, anchor="center")
            table.column("color", width=150, anchor="center")
            table.column("confidence", width=120, anchor="center")
            for row in digit_rows:
                table.insert("", "end", values=row)
            table.pack(fill="x", padx=22, pady=(0, 12))
        else:
            tk.Label(dialog, text=f"Texto de referencia: {detail}", bg=WHITE, fg=TEXT,
                     wraplength=490, justify="left", padx=12, pady=10).pack(fill="x", padx=22, pady=(0, 12))

        form = tk.Frame(dialog, bg=BACKGROUND)
        form.pack(fill="x", padx=22)
        tk.Label(form, text="Importe", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        amount_var = tk.StringVar(value=amount)
        amount_entry = ttk.Entry(form, textvariable=amount_var, width=28, font=("Segoe UI", 12))
        amount_entry.grid(row=0, column=1, sticky="ew", padx=(14, 0), pady=5)
        tk.Label(form, text="Fecha (AAAA-MM-DD)", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        date_var = tk.StringVar(
            value="" if date_required else expense_date or date.today().isoformat()
        )
        date_entry = ttk.Entry(form, textvariable=date_var, width=28, font=("Segoe UI", 11))
        date_entry.grid(row=1, column=1, sticky="ew", padx=(14, 0), pady=5)
        tk.Label(form, text="Categoría", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        category_var = tk.StringVar(value="comida")
        category_box = ttk.Combobox(form, textvariable=category_var,
                                    values=("comida", "transporte", "hogar", "otros"),
                                    state="normal", width=25)
        category_box.grid(row=2, column=1, sticky="ew", padx=(14, 0), pady=5)
        form.grid_columnconfigure(1, weight=1)

        buttons = tk.Frame(dialog, bg=BACKGROUND)
        buttons.pack(fill="x", padx=22, pady=(18, 20))

        def save() -> None:
            try:
                if date_required and not date_var.get().strip():
                    raise ValueError("Escribe la fecha del documento antes de guardarlo.")
                saved_path = save_expense(
                    amount_var.get().strip(), category_var.get(), self.csv_path,
                    date_var.get().strip(),
                )
            except ValueError as exc:
                messagebox.showerror("Revisa los datos", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.status.set(f"Gasto guardado en {saved_path.name}. Menú principal listo.")
            messagebox.showinfo("Gasto guardado", f"El gasto se añadió a:\n{saved_path}", parent=self.root)

        tk.Button(buttons, text="Guardar gasto", command=save, bg=GREEN, fg=WHITE,
                  relief="flat", padx=16, pady=9, font=("Segoe UI", 10, "bold")).pack(side="right")
        tk.Button(buttons, text="Cancelar", command=dialog.destroy, bg="#e7ebf0", fg=TEXT,
                  relief="flat", padx=16, pady=9, font=("Segoe UI", 10)).pack(side="right", padx=(0, 8))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        amount_entry.focus_set()
        amount_entry.selection_range(0, "end")
        self.root.wait_window(dialog)

    def show_history(self) -> None:
        if not self.csv_path.is_file():
            messagebox.showinfo("Historial vacío", "Aún no hay gastos guardados.", parent=self.root)
            self.status.set("No hay registros todavía.")
            return
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error) as exc:
            raise ValueError(f"No se pudo leer el historial: {exc}") from exc
        if not rows:
            messagebox.showinfo("Historial vacío", "El archivo no contiene registros.", parent=self.root)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Gastos guardados")
        dialog.transient(self.root)
        dialog.geometry("680x430")
        dialog.configure(bg=BACKGROUND)
        tk.Label(dialog, text="Historial de gastos", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(16, 10))
        frame = tk.Frame(dialog, bg=BACKGROUND)
        frame.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        tree = ttk.Treeview(frame, columns=("fecha", "importe", "categoria"), show="headings")
        for key, title, width in (("fecha", "Fecha", 170), ("importe", "Importe", 170),
                                  ("categoria", "Categoría", 230)):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for row in reversed(rows):
            tree.insert("", "end", values=(row.get("fecha", ""), row.get("importe", ""), row.get("categoria", "")))
        tk.Button(dialog, text="Volver al menú", command=dialog.destroy, bg=BLUE, fg=WHITE,
                  relief="flat", padx=15, pady=8, font=("Segoe UI", 10, "bold")).pack(anchor="e", padx=18, pady=(0, 15))
        self.root.wait_window(dialog)
        self.status.set("Historial cerrado. Menú principal listo.")

    def _confirm_exit(self) -> None:
        if messagebox.askyesno("Salir", "¿Quieres cerrar la aplicación?", parent=self.root):
            self.root.destroy()


def run_gui(model_path: Path, csv_path: Path) -> None:
    root = tk.Tk()
    ExpenseReaderGUI(root, model_path, csv_path)
    root.mainloop()

