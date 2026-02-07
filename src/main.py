import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finance.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK(kind IN ('Gasto', 'Ingreso'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movement_date TEXT NOT NULL,
                amount REAL NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('Gasto', 'Ingreso')),
                concept_id INTEGER,
                notes TEXT,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
            """
        )


def parse_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("La fecha debe tener formato YYYY-MM-DD")
    return parsed.isoformat()


def parse_amount(value: str) -> float:
    try:
        amount = float(value)
    except ValueError:
        raise ValueError("El monto debe ser un número")
    if amount <= 0:
        raise ValueError("El monto debe ser mayor a 0")
    return amount


class FinanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Organizador Financiero")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.balance_var = tk.StringVar(value="Balance: 0")
        self.income_var = tk.StringVar(value="Ingresos: 0")
        self.expense_var = tk.StringVar(value="Gastos: 0")

        self.filter_start_var = tk.StringVar()
        self.filter_end_var = tk.StringVar(value=date.today().isoformat())

        self.concept_name_var = tk.StringVar()
        self.concept_kind_var = tk.StringVar(value="Gasto")

        self.movement_date_var = tk.StringVar(value=date.today().isoformat())
        self.movement_amount_var = tk.StringVar()
        self.movement_kind_var = tk.StringVar(value="Gasto")
        self.movement_concept_var = tk.StringVar()
        self.movement_notes_var = tk.StringVar()

        self._build_ui()
        self.refresh_concepts()
        self.refresh_movements()
        self.refresh_balance()

    def _build_ui(self):
        balance_frame = ttk.LabelFrame(self, text="Resumen")
        balance_frame.pack(fill="x", padx=16, pady=10)

        ttk.Label(balance_frame, textvariable=self.balance_var, font=("Arial", 14, "bold")).pack(
            side="left", padx=12, pady=6
        )
        ttk.Label(balance_frame, textvariable=self.income_var, foreground="green").pack(
            side="left", padx=12
        )
        ttk.Label(balance_frame, textvariable=self.expense_var, foreground="red").pack(
            side="left", padx=12
        )

        filter_frame = ttk.LabelFrame(self, text="Filtrar por fechas")
        filter_frame.pack(fill="x", padx=16, pady=8)

        ttk.Label(filter_frame, text="Desde (YYYY-MM-DD):").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(filter_frame, textvariable=self.filter_start_var, width=15).grid(
            row=0, column=1, padx=6, pady=6
        )
        ttk.Label(filter_frame, text="Hasta (YYYY-MM-DD):").grid(row=0, column=2, padx=8, pady=6, sticky="w")
        ttk.Entry(filter_frame, textvariable=self.filter_end_var, width=15).grid(
            row=0, column=3, padx=6, pady=6
        )
        ttk.Button(filter_frame, text="Aplicar", command=self.refresh_movements).grid(
            row=0, column=4, padx=8, pady=6
        )

        movement_frame = ttk.LabelFrame(self, text="Movimientos")
        movement_frame.pack(fill="both", expand=True, padx=16, pady=8)

        columns = ("fecha", "concepto", "tipo", "monto", "notas")
        self.movement_table = ttk.Treeview(movement_frame, columns=columns, show="headings")
        self.movement_table.heading("fecha", text="Fecha")
        self.movement_table.heading("concepto", text="Concepto")
        self.movement_table.heading("tipo", text="Tipo")
        self.movement_table.heading("monto", text="Monto")
        self.movement_table.heading("notas", text="Notas")
        self.movement_table.column("fecha", width=100)
        self.movement_table.column("concepto", width=150)
        self.movement_table.column("tipo", width=80)
        self.movement_table.column("monto", width=100, anchor="e")
        self.movement_table.column("notas", width=300)
        self.movement_table.pack(fill="both", expand=True, padx=8, pady=8)

        form_container = ttk.Frame(self)
        form_container.pack(fill="x", padx=16, pady=10)

        concept_frame = ttk.LabelFrame(form_container, text="Crear concepto")
        concept_frame.pack(side="left", fill="both", expand=True, padx=8)

        ttk.Label(concept_frame, text="Nombre:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(concept_frame, textvariable=self.concept_name_var).grid(
            row=0, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Label(concept_frame, text="Tipo:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        ttk.Combobox(
            concept_frame,
            textvariable=self.concept_kind_var,
            values=["Gasto", "Ingreso"],
            state="readonly",
        ).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(concept_frame, text="Guardar concepto", command=self.add_concept).grid(
            row=2, column=0, columnspan=2, padx=6, pady=10
        )
        concept_frame.columnconfigure(1, weight=1)

        movement_form = ttk.LabelFrame(form_container, text="Agregar movimiento")
        movement_form.pack(side="left", fill="both", expand=True, padx=8)

        ttk.Label(movement_form, text="Fecha:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(movement_form, textvariable=self.movement_date_var).grid(
            row=0, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Label(movement_form, text="Monto:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(movement_form, textvariable=self.movement_amount_var).grid(
            row=1, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Label(movement_form, text="Tipo:").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        ttk.Combobox(
            movement_form,
            textvariable=self.movement_kind_var,
            values=["Gasto", "Ingreso"],
            state="readonly",
        ).grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        ttk.Label(movement_form, text="Concepto:").grid(row=3, column=0, padx=6, pady=6, sticky="w")
        self.movement_concept_combo = ttk.Combobox(
            movement_form,
            textvariable=self.movement_concept_var,
            values=[],
            state="readonly",
        )
        self.movement_concept_combo.grid(row=3, column=1, padx=6, pady=6, sticky="ew")
        ttk.Label(movement_form, text="Notas:").grid(row=4, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(movement_form, textvariable=self.movement_notes_var).grid(
            row=4, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(movement_form, text="Guardar movimiento", command=self.add_movement).grid(
            row=5, column=0, columnspan=2, padx=6, pady=10
        )
        movement_form.columnconfigure(1, weight=1)

        self.movement_kind_var.trace_add("write", lambda *_: self.refresh_concepts())

    def refresh_concepts(self):
        kind = self.movement_kind_var.get()
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT name FROM concepts WHERE kind = ? ORDER BY name", (kind,)
            )
            concepts = [row[0] for row in cursor.fetchall()]
        self.movement_concept_combo["values"] = concepts
        if concepts:
            if self.movement_concept_var.get() not in concepts:
                self.movement_concept_var.set(concepts[0])
        else:
            self.movement_concept_var.set("")

    def add_concept(self):
        name = self.concept_name_var.get().strip()
        kind = self.concept_kind_var.get()
        if not name:
            messagebox.showerror("Error", "El nombre del concepto es obligatorio")
            return
        with sqlite3.connect(DB_PATH) as conn:
            try:
                conn.execute(
                    "INSERT INTO concepts (name, kind) VALUES (?, ?)", (name, kind)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "El concepto ya existe")
                return
        self.concept_name_var.set("")
        self.refresh_concepts()
        messagebox.showinfo("Listo", "Concepto guardado")

    def add_movement(self):
        try:
            movement_date = parse_date(self.movement_date_var.get().strip())
            amount = parse_amount(self.movement_amount_var.get().strip())
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return
        kind = self.movement_kind_var.get()
        concept_name = self.movement_concept_var.get().strip()
        notes = self.movement_notes_var.get().strip()
        concept_id = None
        if concept_name:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute(
                    "SELECT id FROM concepts WHERE name = ? AND kind = ?",
                    (concept_name, kind),
                )
                row = cursor.fetchone()
                if row:
                    concept_id = row[0]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO movements (movement_date, amount, kind, concept_id, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (movement_date, amount, kind, concept_id, notes),
            )
            conn.commit()
        self.movement_amount_var.set("")
        self.movement_notes_var.set("")
        self.refresh_movements()
        self.refresh_balance()
        messagebox.showinfo("Listo", "Movimiento guardado")

    def refresh_movements(self):
        start_date = self.filter_start_var.get().strip()
        end_date = self.filter_end_var.get().strip()
        params = []
        query = (
            "SELECT m.movement_date, c.name, m.kind, m.amount, m.notes "
            "FROM movements m LEFT JOIN concepts c ON m.concept_id = c.id"
        )
        conditions = []
        if start_date:
            try:
                start_date = parse_date(start_date)
                conditions.append("m.movement_date >= ?")
                params.append(start_date)
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return
        if end_date:
            try:
                end_date = parse_date(end_date)
                conditions.append("m.movement_date <= ?")
                params.append(end_date)
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY m.movement_date DESC"

        for row in self.movement_table.get_children():
            self.movement_table.delete(row)

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query, params)
            for movement_date, concept_name, kind, amount, notes in cursor.fetchall():
                concept_name = concept_name or "(Sin concepto)"
                self.movement_table.insert(
                    "",
                    tk.END,
                    values=(movement_date, concept_name, kind, f"{amount:,.2f}", notes or ""),
                )

    def refresh_balance(self):
        with sqlite3.connect(DB_PATH) as conn:
            income = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM movements WHERE kind = 'Ingreso'"
            ).fetchone()[0]
            expense = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM movements WHERE kind = 'Gasto'"
            ).fetchone()[0]
        balance = income - expense
        self.income_var.set(f"Ingresos: {income:,.2f}")
        self.expense_var.set(f"Gastos: {expense:,.2f}")
        self.balance_var.set(f"Balance: {balance:,.2f}")


def main():
    init_db()
    app = FinanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
