"""
comparacao_folds.py
Comparação dos três modelos sobre as MESMAS folds da validação cruzada repetida
(5 folds x 10 repetições = 50 avaliações por modelo, só no treino), com testes
pareados entre modelos.

RESSALVA IMPORTANTE: as 50 avaliações NÃO são independentes. Cada fold de teste
partilha doentes com os de outras folds/repetições, e os treinos sobrepõem-se
(Dietterich 1998; Nadeau & Bengio 2003). Um teste que trate as 50 como
independentes dá p-valores demasiado otimistas. Por isso reportam-se:
  - Wilcoxon pareado (ingénuo, a título indicativo; otimista), e
  - t pareado com a correção de Nadeau & Bengio (variância corrigida pela
    sobreposição dos treinos), que é o valor a privilegiar.
Mesmo assim, são 3 comparações (ajuste de Bonferroni) sobre um dataset pequeno
e fácil: os p-valores servem para orientar, não para provar.

Usa o mesmo CV e os mesmos pipelines do comparar_modelos.py.

Saídas:
  results/tables/folds_scores.csv         scores por fold (50 por modelo e métrica)
  results/tables/comparacao_folds.csv     diferenças e p-valores por par de modelos
  results/figures/fig_comparacao_folds.png
"""
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from comparar_modelos import N_FOLDS, N_REPETICOES, SEED, construir_pipelines
from config import CSV, FIGURAS, TABELAS
from Prep_Data import dividir, preparar

METRICAS = {"sensibilidade": "recall", "auroc": "roc_auc", "f1": "f1", "precisao": "precision"}
LANG = "en"   # idioma das legendas da figura: "en" (paper) ou "pt"
T = {"en": dict(sens="Sensitivity", auc="AUROC",
                tit="Per-fold scores (5x10 repeated CV, training set)",
                nota="Folds are not independent (overlapping training sets)."),
     "pt": dict(sens="Sensibilidade", auc="AUROC",
                tit="Scores por fold (CV 5x10 repetida, treino)",
                nota="As folds não são independentes (treinos sobrepostos).")}[LANG]


def t_corrigido(d, n_treino, n_teste):
    """t pareado com a correção de Nadeau & Bengio (2003):
    var corrigida = (1/k + n_teste/n_treino) * var(d). Devolve (t, p bilateral)."""
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = np.mean(d) / np.sqrt((1 / k + n_teste / n_treino) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


def main():
    X, y = preparar(CSV)
    X_tr, X_te, y_tr, y_te = dividir(X, y)

    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=SEED)
    n_total = len(y_tr)
    n_teste = n_total // N_FOLDS                 # tamanho de cada fold de teste
    n_treino = n_total - n_teste

    pipelines = construir_pipelines(X_tr)
    scores = {}
    for nome, pipe in pipelines.items():
        # mesmo objeto cv e mesma seed => os 3 modelos veem exatamente as mesmas folds
        r = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1, scoring=list(METRICAS.values()))
        scores[nome] = {m: r[f"test_{s}"] for m, s in METRICAS.items()}

    # ---- scores por fold
    linhas = []
    for nome, d in scores.items():
        for i in range(len(d["sensibilidade"])):
            linhas.append({"modelo": nome, "fold": i, "repeticao": i // N_FOLDS,
                           **{m: d[m][i] for m in METRICAS}})
    pd.DataFrame(linhas).round(6).to_csv(TABELAS / "folds_scores.csv", index=False,
                                         encoding="utf-8-sig")

    # ---- testes pareados (todos os pares, métricas principais)
    comp = []
    for m in ["sensibilidade", "auroc"]:
        pares = list(combinations(scores, 2))
        for a, b in pares:
            d = scores[a][m] - scores[b][m]
            if np.all(d == 0):                    # modelos idênticos nesta métrica
                p_w = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_w = stats.wilcoxon(scores[a][m], scores[b][m]).pvalue
            t, p_c = t_corrigido(d, n_treino, n_teste)
            comp.append({
                "metrica": m, "modelo_A": a, "modelo_B": b,
                "media_A": scores[a][m].mean(), "media_B": scores[b][m].mean(),
                "dif_media_A_menos_B": d.mean(),
                "folds_A_melhor": int((d > 0).sum()), "folds_B_melhor": int((d < 0).sum()),
                "folds_empate": int((d == 0).sum()),
                "p_wilcoxon_ingenuo": p_w,
                "p_nadeau_bengio": p_c,
                "p_nadeau_bengio_bonferroni": min(1.0, p_c * len(pares)) if not np.isnan(p_c) else np.nan,
            })
    comp = pd.DataFrame(comp)
    comp.round(6).to_csv(TABELAS / "comparacao_folds.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(f"{len(scores)} modelos x {N_FOLDS * N_REPETICOES} folds (treino={n_treino}, "
          f"teste por fold={n_teste})\n")
    print(comp[["metrica", "modelo_A", "modelo_B", "dif_media_A_menos_B", "folds_A_melhor",
                "folds_B_melhor", "folds_empate", "p_wilcoxon_ingenuo", "p_nadeau_bengio",
                "p_nadeau_bengio_bonferroni"]].round(4).to_string(index=False))
    print("\nRESSALVA: as folds não são independentes; o p de Nadeau-Bengio corrige isso,"
          " o Wilcoxon é otimista.")

    # ---- figura: scores por fold
    cores = ["#7f8c8d", "#2e6fa7", "#c0392b"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a_, m, rot in [(ax[0], "sensibilidade", T["sens"]), (ax[1], "auroc", T["auc"])]:
        dados = [scores[n][m] for n in scores]
        bp = a_.boxplot(dados, widths=0.5, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], cores):
            patch.set(facecolor=c, alpha=0.25)
        rng = np.random.default_rng(0)
        for i, (d, c) in enumerate(zip(dados, cores), 1):
            a_.scatter(rng.normal(i, 0.06, len(d)), d, s=8, color=c, alpha=0.6)
        a_.set_xticks(range(1, len(scores) + 1),
                      [n.replace(" (baseline)", "\n(baseline)").replace("Regressão Logística",
                       "Logistic Regression" if LANG == "en" else "Regressão Logística")
                       for n in scores], fontsize=8)
        a_.set_ylabel(rot); a_.grid(axis="y", alpha=0.3)
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(T["tit"], fontsize=10)
    fig.text(0.5, 0.005, T["nota"], ha="center", fontsize=7, color="#7f8c8d")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURAS / "fig_comparacao_folds.png", dpi=300)
    plt.close(fig)
    print(f"\nGuardado em {TABELAS} e {FIGURAS}.")


if __name__ == "__main__":
    main()