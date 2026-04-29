"""Recria o banco de dados do zero com o schema correto."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from database import engine, Base
import models

print("Apagando banco antigo...")
Base.metadata.drop_all(bind=engine)
print("Criando banco com schema correto...")
Base.metadata.create_all(bind=engine)
print("OK - banco recriado!")
