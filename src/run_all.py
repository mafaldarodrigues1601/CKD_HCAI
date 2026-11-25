"""
run_all.py
Um só comando que reproduz os resultados do projeto, por ordem:
  1. comparar_modelos.py   comparação dos 3 modelos (CV repetida) + teste -> results/tables
  2. limiar_calibracao.py  calibração + limiar -> results/models/modelo_ckd_final.joblib
  3. graficos_finais.py    ROC, matriz de confusão, tabela CV, cenários -> results/figures
  4. active_learning_ckd.py               incerteza vs aleatório, split fixo
  5. active_learning_ckd.py --vary-split  o mesmo com o split a variar (robustez)
  Também (antes do active learning): explicar_shap.py (SHAP sobre o modelo final) e
  testar_casos.py (casos de exemplo: claramente DRC, limite e saudável).

Uso (a partir da raiz do projeto):
    python src/run_all.py                        # tudo (cada active learning demora ~1-2 min)
    python src/run_all.py --sem-active-learning  # só 1-3
Pára no primeiro erro.
"""
import subprocess
import sys
import time
from pathlib import Path

from config import CSV

SRC = Path(__file__).resolve().parent

PASSOS = [
    ("Comparação dos modelos", ["comparar_modelos.py"]),
    ("Calibração e limiar", ["limiar_calibracao.py"]),
    ("Gráficos finais", ["graficos_finais.py"]),
    ("Comparação sobre as folds", ["comparacao_folds.py"]),
    ("Explicações SHAP", ["explicar_shap.py"]),
    ("Casos de exemplo", ["testar_casos.py"]),
]
AL = ("Active learning", ["active_learning_ckd.py"])
AL_ROBUSTEZ = ("Active learning (split variável)", ["active_learning_ckd.py", "--vary-split"])


def main():
    if not CSV.exists():
        sys.exit(f"Não encontrei o dataset em {CSV}. Põe o ckd_uci.csv na pasta data/.")

    passos = list(PASSOS)
    if "--sem-active-learning" not in sys.argv:
        passos += [AL, AL_ROBUSTEZ]

    t0 = time.time()
    for i, (nome, cmd) in enumerate(passos, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(passos)}] {nome}  ({' '.join(cmd)})\n{'=' * 70}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(SRC / cmd[0]), *cmd[1:]], cwd=SRC.parent)
        if r.returncode != 0:
            sys.exit(f"\nFalhou em '{nome}' (código {r.returncode}). Resultados parciais em results/.")
        print(f"-> concluído em {time.time() - t:.0f}s", flush=True)

    print(f"\nTudo concluído em {time.time() - t0:.0f}s. Resultados em results/tables, "
          f"results/figures e results/models.")


if __name__ == "__main__":
    main()