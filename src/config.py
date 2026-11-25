"""Caminhos do projeto (um só sítio). Todos os scripts importam daqui,
por isso podem ser corridos de qualquer pasta."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "data" / "ckd_uci.csv"

TABELAS = RAIZ / "results" / "tables"
FIGURAS = RAIZ / "results" / "figures"
MODELOS = RAIZ / "results" / "models"
for pasta in (TABELAS, FIGURAS, MODELOS):
    pasta.mkdir(parents=True, exist_ok=True)
