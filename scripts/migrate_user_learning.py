from __future__ import annotations
"""
Migração simples para bancos SQLite já existentes.

Execute uma vez na raiz do projeto, depois de substituir os arquivos:
python scripts/migrate_user_learning.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("descritoria.db")

COLUNAS = {
    "perfil_aprendizado_ia": "TEXT DEFAULT ''",
    "preferencias_extraidas": "TEXT DEFAULT '{}'",
    "ultima_atualizacao_perfil": "DATETIME",
}


def coluna_existe(cursor, tabela: str, coluna: str) -> bool:
    cursor.execute(f"PRAGMA table_info({tabela})")
    return any(row[1] == coluna for row in cursor.fetchall())


def main():
    if not DB_PATH.exists():
        print("Banco descritoria.db não encontrado. Nada a migrar.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for coluna, ddl in COLUNAS.items():
        if not coluna_existe(cur, "usuarios", coluna):
            print(f"Adicionando coluna usuarios.{coluna}")
            cur.execute(f"ALTER TABLE usuarios ADD COLUMN {coluna} {ddl}")
        else:
            print(f"Coluna usuarios.{coluna} já existe")

    conn.commit()
    conn.close()
    print("Migração concluída.")


if __name__ == "__main__":
    main()
