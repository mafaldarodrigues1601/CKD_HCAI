"""
active_learning_ckd.py
Active learning (amostragem por incerteza vs amostragem aleatória) no dataset CKD,
seguindo a secção Evaluation da proposta.

Uso (a partir da raiz do projeto):
    python src/active_learning_ckd.py
    python src/active_learning_ckd.py --vary-split   # robustez: o split muda com a seed
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import recall_score, roc_auc_score

from comparar_modelos import SEED, construir_pipelines
from config import CSV, FIGURAS, TABELAS
from Prep_Data import dividir, preparar

# ---------------- configuração ----------------
VARY_SPLIT = "--vary-split" in sys.argv
SUFFIX = "_varysplit" if VARY_SPLIT else ""
N_SEEDS = 20
SPLIT_SEED = SEED        # 42: mesmo split do resto do projeto
INIT_SIZE = 10           # casos rotulados no início (5 CKD + 5 sem CKD)
BATCH = 1                # rótulos pedidos por ronda
MAX_LABELS = 100         # orçamento máximo de rótulos
TARGET_SENS = 0.95       # sensibilidade alvo para "labels necessários"
THRESHOLD = 0.5          # limiar de decisão durante o ciclo
MODELO = "Regressão Logística (baseline)"   # modelo escolhido na secção Methods


def make_model(X):
    """Mesmo pipeline do comparar_modelos.py (imputação só nos casos rotulados)."""
    return construir_pipelines(X)[MODELO]


# ---------------- ciclo de active learning ----------------
def least_confidence(proba):
    """1 - max prob. Mais incerto = maior valor."""
    return 1 - proba.max(axis=1)


def run_strategy(strategy, X_tr, y_tr, X_te, y_te, init_idx, seed):
    rng = np.random.default_rng(seed)
    labeled = list(init_idx)
    pool = [i for i in range(len(y_tr)) if i not in set(labeled)]
    n_labels, sens, auc = [], [], []
    queried = []
    while True:
        model = make_model(X_tr).fit(X_tr.iloc[labeled], y_tr[labeled])
        p_te = model.predict_proba(X_te)[:, 1]
        n_labels.append(len(labeled))
        sens.append(recall_score(y_te, p_te >= THRESHOLD))
        auc.append(roc_auc_score(y_te, p_te))
        if len(labeled) >= MAX_LABELS or len(pool) < BATCH:
            break
        if strategy == "uncertainty":
            u = least_confidence(model.predict_proba(X_tr.iloc[pool]))
            chosen = [pool[i] for i in np.argsort(-u)[:BATCH]]
        else:
            chosen = list(rng.choice(pool, size=BATCH, replace=False))
        queried += [y_tr[i] for i in chosen]
        labeled += chosen
        pool = [i for i in pool if i not in set(chosen)]
    return (np.array(n_labels), np.array(sens), np.array(auc),
            float(np.mean(queried)) if queried else np.nan)


_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x e 2.x


def labels_to_target(n, s, target=TARGET_SENS):
    hit = np.where(s >= target)[0]
    return n[hit[0]] if len(hit) else np.nan


def main():
    X, y = preparar(CSV)
    y = y.to_numpy()
    results = {"uncertainty": [], "random": []}
    ref_sens = ref_auc = None
    for seed in range(N_SEEDS):
        split_seed = seed if VARY_SPLIT else SPLIT_SEED
        X_tr, X_te, y_tr, y_te = dividir(X, y, seed=split_seed)
        X_tr = X_tr.reset_index(drop=True)
        if seed == 0:
            print(f"treino {len(y_tr)} ({y_tr.sum()} CKD) | teste {len(y_te)} ({y_te.sum()} CKD)"
                  f" | split {'varia com a seed' if VARY_SPLIT else f'fixo ({SPLIT_SEED})'}")
            if not VARY_SPLIT:
                ref = make_model(X_tr).fit(X_tr, y_tr)               # referência: todo o treino
                p_ref = ref.predict_proba(X_te)[:, 1]
                ref_sens = recall_score(y_te, p_ref >= THRESHOLD)
                ref_auc = roc_auc_score(y_te, p_ref)
                print(f"referência ({len(y_tr)} labels): sens {ref_sens:.3f}, AUROC {ref_auc:.3f}")
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
        init = list(rng.choice(pos, INIT_SIZE // 2, replace=False)) + \
               list(rng.choice(neg, INIT_SIZE // 2, replace=False))
        for s_ in results:
            results[s_].append(run_strategy(s_, X_tr, y_tr, X_te, y_te, init, seed))
    L = min(len(r[0]) for s_ in results for r in results[s_])
    n = results["uncertainty"][0][0][:L]
    sens = {s_: np.array([r[1][:L] for r in results[s_]]) for s_ in results}
    auc = {s_: np.array([r[2][:L] for r in results[s_]]) for s_ in results}
    qpos = {s_: np.array([r[3] for r in results[s_]]) for s_ in results}
    names = {"uncertainty": "Uncertainty sampling", "random": "Random sampling"}
    colors = {"uncertainty": "tab:blue", "random": "tab:orange"}

    # ---- figura
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, data, ylab, ref_v in [(axes[0], sens, "Sensitivity", ref_sens),
                                  (axes[1], auc, "AUROC", ref_auc)]:
        for s_ in data:
            m, sd = data[s_].mean(0), data[s_].std(0)
            ax.plot(n, m, label=names[s_], color=colors[s_])
            ax.fill_between(n, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                            alpha=0.2, color=colors[s_])
        if ref_v is not None:
            ax.axhline(ref_v, ls=":", c="gray", lw=1, label="All training labels")
        ax.set_xlabel("Number of labels"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[0].axhline(TARGET_SENS, ls="--", c="gray", lw=0.8)
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / f"fig_active_learning{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # ---- AULC normalizada + Wilcoxon pareado
    resumo = []
    print("\n--- AULC normalizada (média ± dp) e Wilcoxon pareado ---")
    for label, data in [("sensibilidade", sens), ("AUROC", auc)]:
        a = {s_: _trapz(data[s_], n, axis=1) / (n[-1] - n[0]) for s_ in data}
        stat, p = wilcoxon(a["uncertainty"], a["random"])
        wins = int((a["uncertainty"] > a["random"]).sum())
        print(f"{label}: incerteza {a['uncertainty'].mean():.3f} ± {a['uncertainty'].std():.3f}"
              f" | aleatório {a['random'].mean():.3f} ± {a['random'].std():.3f}"
              f" | Wilcoxon W={stat:.1f}, p={p:.4g} | seeds com AULC(inc) > AULC(aleat): {wins}/{N_SEEDS}")
        resumo.append({"metrica": label,
                       "aulc_incerteza_media": a["uncertainty"].mean(),
                       "aulc_incerteza_dp": a["uncertainty"].std(),
                       "aulc_aleatorio_media": a["random"].mean(),
                       "aulc_aleatorio_dp": a["random"].std(),
                       "wilcoxon_W": stat, "p_valor": p,
                       "seeds_incerteza_melhor": wins, "n_seeds": N_SEEDS})
    pd.DataFrame(resumo).round(6).to_csv(
        TABELAS / f"resultados_active_learning{SUFFIX}.csv", index=False, encoding="utf-8-sig")

    print(f"\n--- Labels para sensibilidade >= {TARGET_SENS} ---")
    for s_ in sens:
        k = np.array([labels_to_target(n, sens[s_][i]) for i in range(N_SEEDS)])
        print(f"{s_}: mediana {np.nanmedian(k)} (IQR {np.nanpercentile(k, 25)}-"
              f"{np.nanpercentile(k, 75)}), {np.isnan(k).sum()} seeds não atingiram")

    print("\n--- Valores em marcos (média ± dp) ---")
    for m_ in [10, 20, 30, 50, 100]:
        j = np.where(n == m_)[0]
        if len(j):
            j = j[0]
            print(f"{m_:>3} labels | sens inc {sens['uncertainty'][:, j].mean():.3f}±"
                  f"{sens['uncertainty'][:, j].std():.3f} aleat {sens['random'][:, j].mean():.3f}±"
                  f"{sens['random'][:, j].std():.3f} | AUROC inc {auc['uncertainty'][:, j].mean():.3f}±"
                  f"{auc['uncertainty'][:, j].std():.3f} aleat {auc['random'][:, j].mean():.3f}±"
                  f"{auc['random'][:, j].std():.3f}")

    print("\n--- Fração de CKD entre os labels pedidos (após os iniciais) ---")
    for s_ in qpos:
        print(f"{s_}: {qpos[s_].mean():.3f} ± {qpos[s_].std():.3f}")

    pd.DataFrame({"n_labels": n,
                  **{f"sens_mean_{s_}": sens[s_].mean(0) for s_ in sens},
                  **{f"sens_std_{s_}": sens[s_].std(0) for s_ in sens},
                  **{f"auroc_mean_{s_}": auc[s_].mean(0) for s_ in sens},
                  **{f"auroc_std_{s_}": auc[s_].std(0) for s_ in sens}}) \
      .to_csv(TABELAS / f"learning_curves{SUFFIX}.csv", index=False)
    print(f"\nGuardado em {FIGURAS} e {TABELAS}.")


if __name__ == "__main__":
    main()
