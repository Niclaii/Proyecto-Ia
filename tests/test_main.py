import unittest
import sqlite3
from pathlib import Path
import tempfile

import src.main as main


class FinanceAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "finance.db"
        main.DB_PATH = self.db_path

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_tables(self):
        main.init_db()
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertIn("concepts", tables)
        self.assertIn("movements", tables)

    def test_parse_date_valid_and_invalid(self):
        self.assertEqual(main.parse_date("2024-10-05"), "2024-10-05")
        with self.assertRaises(ValueError):
            main.parse_date("05-10-2024")

    def test_parse_amount_valid_and_invalid(self):
        self.assertEqual(main.parse_amount("10"), 10.0)
        with self.assertRaises(ValueError):
            main.parse_amount("abc")
        with self.assertRaises(ValueError):
            main.parse_amount("0")

    def test_insert_movement_with_concept(self):
        main.init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO concepts (name, kind) VALUES (?, ?)",
                ("Salario", "Ingreso"),
            )
            concept_id = conn.execute(
                "SELECT id FROM concepts WHERE name = ?",
                ("Salario",),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO movements (movement_date, amount, kind, concept_id, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("2024-10-05", 1500.0, "Ingreso", concept_id, "Pago mensual"),
            )
            conn.commit()
            total = conn.execute(
                "SELECT COUNT(*) FROM movements"
            ).fetchone()[0]
        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
