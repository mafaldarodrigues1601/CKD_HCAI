"""
limiar_calibracao.py
Passo seguinte ao comparar_modelos.py: calibração + limiar por sensibilidade-alvo.

Tudo é decidido SÓ com o treino (probabilidades fora-da-amostra por validação
cruzada repetida). O teste é usado uma vez, no fim, para reportar.

Ordem:
  1. Calibração: compara Brier (fora-da-amostra, treino) da probabilidade original
     vs calibrada (sigmoide). Usa-se a calibrada só se o Brier melhorar.
  2. Limiar: o MAIOR limiar cuja sensibilidade fora-da-amostra seja >= ALVO_SENS
     (maior limiar = mais especificidade, mantendo a sensibilidade exigida).
  3. Avaliação final no teste: sensibilidade, especificidade, Brier, curva de calibração.

Requer: ter corrido antes o comparar_modelos.py (gera results/tables/resultados_cv.csv).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from comparar_modelos import SEED, construir_pipelines
from config import CSV, FIGURAS, MODELOS, TABELAS
from Prep_Data import dividir, preparar

# --- DEFINIDO À PARTIDA (decisão clínica; justificar no relatório) ---------
ALVO_SENS = 0.98          # sensibilidade mínima exigida
N_REPETICOES = 10         # repetições da CV para as probabilidades fora-da-amostra
# ---------------------------------------------------------------------------

LANG = "en"   # idioma das legendas das figuras: "en" (proposta em inglês) ou "pt"
T = {
    "en": dict(orig="Original", calib="Calibrated (sigmoid)", perfeita="Perfectly calibrated",
               t_oof="Calibration (training set, out-of-fold)", t_teste="Calibration (test set, n={n})",
               x="Predicted probability", y="Observed fraction of CKD",
               t_hist="Predicted probabilities (test set)", x_hist="Predicted probability of CKD",
               y_hist="Number of patients", sem="No CKD (actual)", com="CKD (actual)",
               lim="Threshold {t:.2f}"),
    "pt": dict(orig="Original", calib="Calibrado (sigmoide)", perfeita="Perfeita",
               t_oof="Calibração (treino, fora-da-amostra)", t_teste="Calibração (teste, n={n})",
               x="Probabilidade prevista", y="Fração real de CKD",
               t_hist="Probabilidades no teste", x_hist="Probabilidade de CKD",
               y_hist="Nº de doentes", sem="Sem CKD (real)", com="CKD (real)",
               lim="Limiar {t:.2f}"),
}[LANG]
NOMES_PROB = {"original": T["orig"], "calibrado (sigmoide)": T["calib"]}


def probs_oof(modelo, X, y):
    """Probabilidades fora-da-amostra, empilhadas ao longo das repetições."""
    ps, ys = [], []
    for r in range(N_REPETICOES):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        ps.append(cross_val_predict(clone(modelo), X, y, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1])
        ys.append(y.values)
    return np.concatenate(ps), np.concatenate(ys)


def limiar_para_sensibilidade(prob, y, alvo):
    """Maior limiar t (previsão = prob >= t) com sensibilidade >= alvo."""
    pos = prob[y == 1]
    for t in np.unique(pos)[::-1]:
        if (pos >= t).mean() >= alvo:
            return float(t)
    return 0.0


def sens_espec(prob, y, t):
    pred = prob >= t
    return (pred[y == 1]).mean(), (~pred[y == 0]).mean()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = preparar(CSV)
    X_tr, X_te, y_tr, y_te = dividir(X, y)

    # Modelo escolhido no comparar_modelos.py
    tab = pd.read_csv(TABELAS / "resultados_cv.csv")
    nome = tab.loc[tab["selecionado"], "modelo"].iloc[0]
    base = construir_pipelines(X_tr)[nome]
    print(f"Modelo: {nome}\n")

    # 1. Calibração (decidida no treino)
    candidatos = {
        "original": base,
        "calibrado (sigmoide)": CalibratedClassifierCV(base, method="sigmoid", cv=5),
    }
    oof, brier = {}, {}
    for k, m in candidatos.items():
        p, yy = probs_oof(m, X_tr, y_tr)
        oof[k], brier[k] = p, brier_score_loss(yy, p)
    print("Brier fora-da-amostra (treino, menor = melhor):")
    for k, v in brier.items():
        print(f"  {k:22s}: {v:.4f}")
    escolha = min(brier, key=brier.get)
    print(f"-> Probabilidade usada: {escolha}\n")
    modelo_final = candidatos[escolha]

    # 2. Limiar por sensibilidade-alvo (fora-da-amostra, treino)
    p_oof, y_oof = oof[escolha], np.tile(y_tr.values, N_REPETICOES)
    limiar = limiar_para_sensibilidade(p_oof, y_oof, ALVO_SENS)
    s, e = sens_espec(p_oof, y_oof, limiar)
    s5, e5 = sens_espec(p_oof, y_oof, 0.5)
    print(f"Limiar para sensibilidade >= {ALVO_SENS:.2f}: {limiar:.3f}")
    print(f"  treino fora-da-amostra: sens {s:.3f} | espec {e:.3f}   (limiar 0.5: sens {s5:.3f} | espec {e5:.3f})")

    # Trade-off para o relatório (vários alvos, só treino)
    linhas = []
    for alvo in [0.95, 0.98, 0.99, 1.0]:
        t = limiar_para_sensibilidade(p_oof, y_oof, alvo)
        se, es = sens_espec(p_oof, y_oof, t)
        linhas.append({"sensibilidade_alvo": alvo, "limiar": round(t, 4),
                       "sens_treino_oof": round(se, 4), "espec_treino_oof": round(es, 4)})
    tradeoff = pd.DataFrame(linhas)
    print("\nTrade-off sensibilidade/especificidade (treino, fora-da-amostra):")
    print(tradeoff.to_string(index=False))
    tradeoff.to_csv(TABELAS / "resultados_limiar.csv", index=False, encoding="utf-8-sig")

    # 3. Avaliação final no teste (uma vez)
    modelo_final.fit(X_tr, y_tr)
    p_te = modelo_final.predict_proba(X_te)[:, 1]
    yt = y_te.values
    s_te, e_te = sens_espec(p_te, yt, limiar)
    s_05, e_05 = sens_espec(p_te, yt, 0.5)
    fn = int(((p_te < limiar) & (yt == 1)).sum())
    fp = int(((p_te >= limiar) & (yt == 0)).sum())
    print(f"\nTeste (limiar {limiar:.3f}): sensibilidade {s_te:.3f} | especificidade {e_te:.3f} "
          f"| FN {fn} | FP {fp}")
    print(f"Teste (limiar 0.5):   sensibilidade {s_05:.3f} | especificidade {e_05:.3f}")
    print(f"Brier no teste: {brier_score_loss(yt, p_te):.4f} | AUROC: {roc_auc_score(yt, p_te):.3f}")

    pd.DataFrame([{
        "modelo": nome, "probabilidade": escolha, "alvo_sens": ALVO_SENS,
        "limiar": round(limiar, 4), "sensibilidade_teste": round(s_te, 4),
        "especificidade_teste": round(e_te, 4), "falsos_negativos": fn,
        "falsos_positivos": fp, "brier_teste": round(brier_score_loss(yt, p_te), 4),
        "brier_oof_treino": round(brier[escolha], 4), "auroc_teste": round(roc_auc_score(yt, p_te), 4),
    }]).to_csv(TABELAS / "resultados_teste_limiar.csv", index=False, encoding="utf-8-sig")

    joblib.dump({"modelo": modelo_final, "limiar": limiar, "nome": nome,
                 "probabilidade": escolha}, MODELOS / "modelo_ckd_final.joblib")
    print("\nGuardado: results/models/modelo_ckd_final.joblib (modelo + limiar)")

    # Gráficos
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, p in oof.items():
        fr, md = calibration_curve(np.tile(y_tr.values, N_REPETICOES), p, n_bins=10)
        ax[0].plot(md, fr, "o-", label=f"{NOMES_PROB[k]} (Brier {brier[k]:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", label=T["perfeita"])
    ax[0].set(title=T["t_oof"], xlabel=T["x"], ylabel=T["y"])
    ax[0].legend(loc="upper left", fontsize=8)

    fr, md = calibration_curve(yt, p_te, n_bins=5)
    ax[1].plot(md, fr, "o-", label=f"{NOMES_PROB[escolha]} (Brier {brier_score_loss(yt, p_te):.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", label=T["perfeita"])
    ax[1].set(title=T["t_teste"].format(n=len(yt)), xlabel=T["x"], ylabel=T["y"])
    ax[1].legend(loc="upper left", fontsize=8)

    bins = np.linspace(0, 1, 21)
    ax[2].hist(p_te[yt == 0], bins=bins, alpha=0.7, label=T["sem"])
    ax[2].hist(p_te[yt == 1], bins=bins, alpha=0.7, label=T["com"])
    ax[2].axvline(limiar, color="k", ls="--", label=T["lim"].format(t=limiar))
    ax[2].set(title=T["t_hist"], xlabel=T["x_hist"], ylabel=T["y_hist"])
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "limiar_calibracao.png", dpi=150)
    print("Gráficos guardados em results/figures/limiar_calibracao.png")


if __name__ == "__main__":
    main()