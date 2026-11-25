"""
graficos_finais.py
Figuras e tabelas finais para o relatório:
  1. fig_roc_confusao.png      curva ROC + matriz de confusão (teste, modelo final + limiar)
  2. tabela_cv.png / .csv      tabela da validação cruzada (3 modelos)
  3. fig_cenarios_limitacao.png + resultados_cenarios_rf.csv
                               Random Forest sem os marcadores renais (cenários B-D),
                               apresentado como LIMITAÇÃO

Requer ter corrido antes comparar_modelos.py e limiar_calibracao.py.
Mudar LANG para "pt" para legendas em português.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from comparar_modelos import N_FOLDS, N_REPETICOES, SEED, construir_pipelines
from config import CSV, FIGURAS, MODELOS, TABELAS
from Prep_Data import dividir, preparar

LANG = "en"   # "en" (proposta em inglês) ou "pt"

T = {
    "en": dict(sens="Sensitivity", spec="Specificity", auc="AUROC", f1="F1-score", prec="Precision",
               modelo="Model", roc="ROC curve (test set)", fpr="1 - Specificity (false positive rate)",
               tpr="Sensitivity", acaso="Chance", op="Operating point (threshold {t:.2f})",
               cm="Confusion matrix (threshold {t:.2f})", prev="Predicted", real="Actual",
               classes=["No CKD", "CKD"], tabela="Cross-validation on the training set ({k} folds x {r} repetitions)",
               cen=["A) All variables", "B) Without renal markers\n(sc, bu, al, sg)",
                    "C) Risk factors only", "D) Risk factors\n+ hemoglobin"],
               cen_tit="Random Forest by variable scenario (CV on training set, mean ± SD)",
               cen_nota="Axis starts at 0.5 (AUROC of chance level).",
               modelos={"Regressão Logística (baseline)": "Logistic Regression (baseline)",
                        "Random Forest": "Random Forest", "XGBoost": "XGBoost"}),
    "pt": dict(sens="Sensibilidade", spec="Especificidade", auc="AUROC", f1="F1", prec="Precisão",
               modelo="Modelo", roc="Curva ROC (teste)", fpr="1 - Especificidade (falsos positivos)",
               tpr="Sensibilidade", acaso="Acaso", op="Ponto de operação (limiar {t:.2f})",
               cm="Matriz de confusão (limiar {t:.2f})", prev="Previsto", real="Real",
               classes=["Sem CKD", "CKD"], tabela="Validação cruzada no treino ({k} folds x {r} repetições)",
               cen=["A) Todas as variáveis", "B) Sem marcadores renais\n(sc, bu, al, sg)",
                    "C) Só fatores de risco", "D) Fatores de risco\n+ hemoglobina"],
               cen_tit="Random Forest por cenário de variáveis (CV no treino, média ± DP)",
               cen_nota="O eixo começa em 0,5 (AUROC do acaso).",
               modelos={"Regressão Logística (baseline)": "Regressão Logística (baseline)",
                        "Random Forest": "Random Forest", "XGBoost": "XGBoost"}),
}[LANG]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
AZUL, VERM, CINZA = "#2e6fa7", "#c0392b", "#7f8c8d"

X, y = preparar(CSV)
X_tr, X_te, y_tr, y_te = dividir(X, y)

# ---------------------------------------------------------------- 1. ROC + matriz de confusão
final = joblib.load(MODELOS / "modelo_ckd_final.joblib")
modelo, limiar = final["modelo"], final["limiar"]
p_te = modelo.predict_proba(X_te)[:, 1]
pred = (p_te >= limiar).astype(int)
fpr, tpr, _ = roc_curve(y_te, p_te)
auc = roc_auc_score(y_te, p_te)
sens = ((pred == 1) & (y_te == 1)).sum() / (y_te == 1).sum()
espec = ((pred == 0) & (y_te == 0)).sum() / (y_te == 0).sum()

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.5))
ax[0].plot(fpr, tpr, color=AZUL, lw=2, label=f"AUROC = {auc:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color=CINZA, lw=1, label=T["acaso"])
ax[0].scatter([1 - espec], [sens], color=VERM, zorder=5, label=T["op"].format(t=limiar))
ax[0].set(xlabel=T["fpr"], ylabel=T["tpr"], title=T["roc"], xlim=(-0.02, 1), ylim=(0, 1.03))
ax[0].legend(loc="lower right", fontsize=8)

cm = confusion_matrix(y_te, pred)
ax[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[1].set_xticks([0, 1], T["classes"]); ax[1].set_yticks([0, 1], T["classes"])
ax[1].set(xlabel=T["prev"], ylabel=T["real"], title=T["cm"].format(t=limiar))
ax[1].spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURAS / "fig_roc_confusao.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 2. tabela da validação cruzada
cv_tab = pd.read_csv(TABELAS / "resultados_cv.csv")
linhas = []
for _, r in cv_tab.iterrows():
    linhas.append([T["modelos"].get(r["modelo"], r["modelo"]),
                   f"{np.round(r['recall_media'], 3):.3f} ± {np.round(r['recall_dp'], 3):.3f}",
                   f"{np.round(r['auc'], 3):.3f}", f"{np.round(r['f1'], 3):.3f}",
                   f"{np.round(r['precisao'], 3):.3f}"])  # np.round = mesmo arredondamento do terminal
cab = [T["modelo"], T["sens"] + " (mean ± SD)" if LANG == "en" else T["sens"] + " (média ± DP)",
       T["auc"], T["f1"], T["prec"]]
tabela_final = pd.DataFrame(linhas, columns=cab)
tabela_final.to_csv(TABELAS / "tabela_cv.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8.2, 1.0 + 0.4 * len(tabela_final)))
ax.axis("off")
tb = ax.table(cellText=tabela_final.values, colLabels=cab, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
tb.auto_set_column_width(list(range(len(cab))))
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor("#bbbbbb")
    if i == 0:
        c.set_facecolor("#e8eef5"); c.set_text_props(weight="bold")
    elif bool(cv_tab.loc[i - 1, "selecionado"]):
        c.set_text_props(weight="bold")
ax.set_title(T["tabela"].format(k=N_FOLDS, r=N_REPETICOES), fontsize=10)
fig.tight_layout()
fig.savefig(FIGURAS / "tabela_cv.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 3. limitação: cenários do RF
marcadores = ["sc", "bu", "al", "sg"]                       # definem a CKD
risco = ["age", "bp", "bgr", "htn", "dm", "cad"]
todas = list(X.columns)
cenarios = [todas, [c for c in todas if c not in marcadores], risco, risco + ["hemo"]]

cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=SEED)
scoring = {"sens": "recall", "auc": "roc_auc",
           "spec": make_scorer(recall_score, pos_label=0)}
res = []
for nome, cols in zip(T["cen"], cenarios):
    r = cross_validate(construir_pipelines(X_tr[cols])["Random Forest"],
                       X_tr[cols], y_tr, cv=cv, scoring=scoring, n_jobs=-1)
    linha = {"cenario": nome.replace("\n", " "), "n_variaveis": len(cols)}
    for m in scoring:
        linha[f"{m}_media"], linha[f"{m}_dp"] = r[f"test_{m}"].mean(), r[f"test_{m}"].std()
    res.append(linha)
    print(f"{linha['cenario']:48s} sens {linha['sens_media']:.3f} ± {linha['sens_dp']:.3f} | "
          f"espec {linha['spec_media']:.3f} | AUC {linha['auc_media']:.3f}")
res = pd.DataFrame(res)
res.round(4).to_csv(TABELAS / "resultados_cenarios_rf.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8.2, 4))
larg, x = 0.26, np.arange(len(res))
for k, (m, cor) in enumerate([("sens", VERM), ("spec", AZUL), ("auc", CINZA)]):
    ax.bar(x + (k - 1) * larg, res[f"{m}_media"], larg, yerr=res[f"{m}_dp"], capsize=3,
           color=cor, label=T[m])
ax.set_xticks(x, T["cen"], fontsize=8)
ax.set_ylim(0.5, 1.02)
ax.set_title(T["cen_tit"], fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, frameon=False)
ax.set_xlabel(T["cen_nota"], fontsize=7, color=CINZA, labelpad=2)
ax.xaxis.set_label_position("top")
fig.tight_layout()
fig.savefig(FIGURAS / "fig_cenarios_limitacao.png", dpi=300)
plt.close(fig)
print("\nGuardado em results/figures e results/tables.")
