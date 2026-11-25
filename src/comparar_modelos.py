"""
comparar_modelos.py
Compara Regressão Logística (baseline), Random Forest e XGBoost no dataset CKD,
seguindo a secção Evaluation da proposta.

Usa a função preparar() do Prep_Data.py (limpeza + codificação 0/1 do grupo).
Lê data/ckd_uci.csv e escreve em results/ (ver config.py).

Uso (a partir da raiz do projeto):
    python src/comparar_modelos.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import CSV, MODELOS, TABELAS
from Prep_Data import CATEGORICAS, SEED, dividir, preparar

# --------------------------------------------------------------------------
# CRITÉRIO DE ESCOLHA - definido ANTES de ver resultados (não alterar depois)
#   1. Métrica principal: sensibilidade média na validação cruzada repetida
#      (um CKD não detetado é o erro mais caro).
#   2. Empate: um modelo empata com o melhor se a diferença entre as sensibilidades
#      médias for MENOR que 1 desvio-padrão da sensibilidade do melhor modelo
#      (desvio-padrão sobre todos os folds).
#   3. Desempate: entre os empatados, o mais simples
#      (regressão logística < random forest < XGBoost).
#   4. A validação cruzada usa só o treino; o teste só é usado uma vez,
#      no modelo escolhido.
# --------------------------------------------------------------------------
N_FOLDS, N_REPETICOES = 5, 10          # 50 avaliações por modelo
METRICA_PRINCIPAL = "recall"
ORDEM_SIMPLICIDADE = ["Regressão Logística (baseline)", "Random Forest", "XGBoost"]


# --------------------------------------------------------------------------
# Pipelines: imputação (mediana / moda) ajustada só no treino + modelo
# --------------------------------------------------------------------------
def construir_pipelines(X):
    cat = [c for c in X.columns if c in CATEGORICAS]
    num = [c for c in X.columns if c not in CATEGORICAS]

    def prep(escalar):
        passos_num = [("imp", SimpleImputer(strategy="median"))]
        passos_cat = [("imp", SimpleImputer(strategy="most_frequent"))]
        if escalar:  # só a regressão logística precisa de escala
            passos_num.append(("esc", StandardScaler()))
            passos_cat.append(("esc", StandardScaler()))
        return ColumnTransformer([("num", Pipeline(passos_num), num),
                                  ("cat", Pipeline(passos_cat), cat)])

    return {
        "Regressão Logística (baseline)": Pipeline([
            ("prep", prep(escalar=True)),
            ("modelo", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep(escalar=False)),
            ("modelo", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ]),
        "XGBoost": Pipeline([  # mesmos hiperparâmetros do XGBoost.py
            ("prep", prep(escalar=False)),
            ("modelo", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.3,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=SEED)),
        ]),
    }


def escolher_modelo(tabela_cv):
    """Aplica o critério definido no topo do ficheiro."""
    melhor = tabela_cv["recall_media"].idxmax()
    limiar = tabela_cv.loc[melhor, "recall_media"] - tabela_cv.loc[melhor, "recall_dp"]
    empatados = [m for m in tabela_cv.index if tabela_cv.loc[m, "recall_media"] > limiar
                 or m == melhor]
    return next(m for m in ORDEM_SIMPLICIDADE if m in empatados), empatados


def main():
    X, y = preparar(CSV)

    # Divisão 80/20 estratificada (igual à do XGBoost.py)
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    print(f"Treino: {len(X_tr)} | Teste: {len(X_te)} | % CKD no treino: {y_tr.mean():.1%}\n")

    # Validação cruzada estratificada REPETIDA, SÓ no treino.
    # Mesma semente => os 3 modelos são avaliados exatamente nos mesmos folds.
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES,
                                 random_state=SEED)
    pipelines = construir_pipelines(X_tr)
    linhas = {}
    for nome, pipe in pipelines.items():
        r = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                           scoring=["recall", "roc_auc", "f1", "precision"])
        linhas[nome] = {
            "recall_media": r["test_recall"].mean(), "recall_dp": r["test_recall"].std(),
            "auc": r["test_roc_auc"].mean(), "f1": r["test_f1"].mean(),
            "precisao": r["test_precision"].mean(),
        }
    tabela_cv = pd.DataFrame(linhas).T
    print(f"Validação cruzada ({N_FOLDS} folds x {N_REPETICOES} repetições, só treino):")
    print(tabela_cv.round(3).to_string())

    escolhido, empatados = escolher_modelo(tabela_cv)

    # Tabela comparativa (validação cruzada) em CSV, com o modelo escolhido assinalado
    tabela_csv = tabela_cv.round(6).copy()
    tabela_csv["selecionado"] = tabela_csv.index == escolhido
    tabela_csv.index.name = "modelo"
    tabela_csv.to_csv(TABELAS / "resultados_cv.csv", encoding="utf-8-sig")

    print(f"\nEmpatados com o melhor (diferença < 1 DP): {', '.join(empatados)}")
    print(f"Modelo selecionado (mais simples entre os empatados): {escolhido}")

    # Avaliação ÚNICA no teste, apenas para o modelo selecionado
    modelo = pipelines[escolhido].fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    prob = modelo.predict_proba(X_te)[:, 1]
    print("\nTeste (modelo selecionado):")
    print(f"  Sensibilidade: {recall_score(y_te, pred):.3f}")
    print(f"  Precisão:      {precision_score(y_te, pred):.3f}")
    print(f"  F1:            {f1_score(y_te, pred):.3f}")
    print(f"  Exatidão:      {accuracy_score(y_te, pred):.3f}")
    print(f"  AUROC:         {roc_auc_score(y_te, prob):.3f}")

    # Métricas do teste em CSV
    pd.DataFrame([{
        "modelo": escolhido,
        "sensibilidade": recall_score(y_te, pred),
        "precisao": precision_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "exatidao": accuracy_score(y_te, pred),
        "auroc": roc_auc_score(y_te, prob),
        "n_teste": len(y_te),
    }]).round(4).to_csv(TABELAS / "resultados_teste.csv", index=False, encoding="utf-8-sig")

    # Guardar o pipeline completo (imputação + modelo), treinado só com os 320 de treino
    joblib.dump(modelo, MODELOS / "modelo_ckd.joblib")
    print("Modelo guardado em results/models/modelo_ckd.joblib")


if __name__ == "__main__":
    main()
