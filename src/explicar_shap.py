"""
explicar_shap.py
Explicações SHAP globais e por doente para o modelo final do grupo
(results/models/modelo_ckd_final.joblib, gerado pelo limiar_calibracao.py).

Escolhe o explainer conforme o modelo guardado:
  * Regressão Logística -> shap.LinearExplainer  (contribuições em log-odds)
  * Random Forest       -> shap.TreeExplainer    (contribuições em probabilidade)
  * XGBoost             -> shap.TreeExplainer    (contribuições em log-odds)
  * outro (ex.: modelo calibrado) -> explainer genérico sobre a probabilidade

Requer ter corrido antes: comparar_modelos.py e limiar_calibracao.py.
Uso (a partir da raiz do projeto):
    python src/explicar_shap.py

Gera:
  results/figures/shap_importancia_global.png   média |SHAP| (top 12), marcadores KDIGO a vermelho
  results/figures/shap_dispersao_global.png     um ponto por doente do teste
  results/figures/shap_doente_<id>.png          3 doentes de exemplo
  results/tables/shap_explicacoes_teste.csv     explicação em texto para cada doente do teste
  results/tables/shap_top5_kdigo.csv            quantas das 5 variáveis principais são marcadores KDIGO
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn import base
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import CSV, FIGURAS, MODELOS, TABELAS
from descricoes import DESCRICOES, FEATURES, descrever_valor, nome
from Prep_Data import CATEGORICAS, NUMERICAS, dividir, preparar

LANG = "en"   # idioma das FIGURAS ("en" para o paper, "pt"); o texto para o médico é sempre PT
CAMINHO_MODELO = MODELOS / "modelo_ckd_final.joblib"

# Marcadores de lesão/função renal usados na definição de DRC (KDIGO 2024).
# São as mesmas 4 variáveis retiradas no cenário B do graficos_finais.py.
MARCADORES_KDIGO = {"sc", "bu", "al", "sg"}
NOMES_EN = {"Regressão Logística (baseline)": "Logistic Regression (baseline)",
            "Random Forest": "Random Forest", "XGBoost": "XGBoost"}

T = {
    "en": dict(imp="Mean |SHAP value| ({e})", imp_t="Global feature importance (SHAP), {m}\nred = KDIGO renal marker",
               disp_x="SHAP value ({e}); right = higher CKD risk", disp_t="SHAP values per patient (test set)",
               cor="feature value (low → high)", doente_t="Patient {i} ({r}): predicted probability of CKD {p:.0%}",
               doente_x="SHAP contribution ({e}); red increases risk",
               escala={"log-odds": "log-odds", "probabilidade": "probability"},
               rotulos={"drc": "most certain CKD", "sem_drc": "most certain no CKD", "incerto": "closest to threshold"}),
    "pt": dict(imp="Média de |SHAP| ({e})", imp_t="Importância global (SHAP), {m}\nvermelho = marcador renal KDIGO",
               disp_x="Valor SHAP ({e}); à direita aumenta o risco", disp_t="Valores SHAP por doente (teste)",
               cor="valor da variável (baixo → alto)", doente_t="Doente {i} ({r}): probabilidade de DRC {p:.0%}",
               doente_x="Contribuição SHAP ({e}); vermelho aumenta o risco",
               escala={"log-odds": "log-odds", "probabilidade": "probabilidade"},
               rotulos={"drc": "DRC mais certo", "sem_drc": "sem DRC mais certo", "incerto": "mais perto do limiar"}),
}[LANG]


# ------------------------------------------------------------------
# Dados e modelo do grupo
# ------------------------------------------------------------------
@lru_cache(maxsize=1)
def dados_divididos():
    """A divisão 80/20 única do projeto (Prep_Data.dividir), igual à dos outros scripts."""
    X, y = preparar(CSV)
    return dividir(X, y)


@lru_cache(maxsize=1)
def carregar_modelo():
    """Dicionário guardado pelo limiar_calibracao.py: modelo, limiar, nome, probabilidade."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Não encontrei {CAMINHO_MODELO}. Corra primeiro "
                                "src/comparar_modelos.py e src/limiar_calibracao.py.")
    return joblib.load(CAMINHO_MODELO)


def _imputador(X_tr):
    """Mesma imputação do modelo (mediana / moda, ajustada no treino), para mostrar
    ao médico que valor foi usado quando um dado está em falta."""
    ct = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERICAS),
                            ("cat", SimpleImputer(strategy="most_frequent"), list(CATEGORICAS))])
    return ct.fit(X_tr)


def _pct(p):
    """Percentagem para texto: evita mostrar 100% ou 0% falsos."""
    if p > 0.99:
        return ">99%"
    if p < 0.01:
        return "<1%"
    return f"{p:.0%}"


class Explicador:
    def __init__(self, final, X_tr):
        import shap   # importado aqui: o resto do projeto funciona sem shap

        self.final = final
        self.modelo = final["modelo"]
        self.imputador = _imputador(X_tr)
        clf = self.modelo[-1] if isinstance(self.modelo, Pipeline) else None
        tipo = type(clf).__name__

        if tipo == "LogisticRegression":
            self.tipo, self.escala = "linear", "log-odds"
            self.transformar = self.modelo[:-1].transform
            self.explainer = shap.LinearExplainer(clf, self.transformar(X_tr))
        elif tipo in ("RandomForestClassifier", "ExtraTreesClassifier"):
            self.tipo, self.escala = "arvore", "probabilidade"
            self.transformar = self.modelo[:-1].transform
            self.explainer = shap.TreeExplainer(clf)
        elif tipo == "XGBClassifier":
            self.tipo, self.escala = "arvore", "log-odds"
            self.transformar = self.modelo[:-1].transform
            self.explainer = shap.TreeExplainer(clf)
        else:   # ex.: CalibratedClassifierCV -> explica diretamente a probabilidade final
            self.tipo, self.escala = "generico", "probabilidade"
            fundo = self.imputador.transform(X_tr)
            self.explainer = shap.Explainer(self._prob_de_array,
                                            shap.maskers.Independent(fundo, max_samples=100))

    def _prob_de_array(self, Z):
        return self.modelo.predict_proba(pd.DataFrame(Z, columns=FEATURES))[:, 1]

    @staticmethod
    def _so_classe_drc(valores, base):
        """Normaliza os formatos do shap para ficar só com a classe DRC (1)."""
        if isinstance(valores, list):          # versões antigas: [classe0, classe1]
            valores = valores[1]
        valores = np.asarray(valores)
        if valores.ndim == 3:                  # versões novas: (n, variáveis, classes)
            valores = valores[:, :, 1]
        base = np.atleast_1d(np.asarray(base, dtype=float))
        return valores, float(base[-1] if base.size == 2 else base[0])

    def calcular(self, X):
        """(valores SHAP n x 24, valor base) para a classe DRC."""
        X = X[FEATURES]
        if self.tipo == "generico":
            exp = self.explainer(self.imputador.transform(X))
            return np.asarray(exp.values), float(np.mean(exp.base_values))
        valores = self.explainer.shap_values(self.transformar(X))
        return self._so_classe_drc(valores, self.explainer.expected_value)

    def saida_modelo(self, X):
        """Saída do modelo na escala do SHAP (para verificar a aditividade)."""
        if self.escala == "probabilidade":
            return self.modelo.predict_proba(X[FEATURES])[:, 1]
        clf, Z = self.modelo[-1], self.transformar(X[FEATURES])
        if type(clf).__name__ == "XGBClassifier":
            return clf.predict(Z, output_margin=True)
        return clf.decision_function(Z)

    def explicar_doente(self, X_linha, n_fatores=5):
        """Explicação de 1 doente (DataFrame com 1 linha e as 24 variáveis; NaN = em falta)."""
        valores, base = self.calcular(X_linha)
        valores = valores[0]
        usados = pd.Series(self.imputador.transform(X_linha[FEATURES])[0], index=FEATURES)
        em_falta = [c for c in FEATURES if pd.isna(X_linha.iloc[0][c])]
        prob = float(self.modelo.predict_proba(X_linha[FEATURES])[0, 1])

        fatores = []
        for i in np.argsort(-np.abs(valores))[:n_fatores]:
            var = FEATURES[i]
            fatores.append({
                "variavel": var,
                "nome": nome(var),
                "valor_usado": float(usados[var]),
                "em_falta": var in em_falta,
                "descricao": descrever_valor(var, usados[var])
                             + (" [em falta: valor típico usado]" if var in em_falta else ""),
                "contribuicao": round(float(valores[i]), 4),
                "efeito": "aumenta o risco" if valores[i] > 0 else "diminui o risco",
            })
        return {
            "probabilidade_drc": round(prob, 4),
            "limiar": round(float(self.final["limiar"]), 4),
            "valor_base": round(base, 4),
            "escala": self.escala,
            "fatores_principais": fatores,
            "variaveis_em_falta": em_falta,
            "texto": texto_explicacao(prob, base, fatores, self.escala, em_falta),
        }


def texto_explicacao(prob, base, fatores, escala, em_falta=()):
    """Explicação em linguagem simples (português) para o médico."""
    def lista(fs):
        if escala == "probabilidade":
            return "; ".join(f"{f['descricao']} ({f['contribuicao'] * 100:+.0f} p.p.)" for f in fs)
        return "; ".join(f["descricao"] for f in fs)

    sobe = [f for f in fatores if f["contribuicao"] > 0]
    desce = [f for f in fatores if f["contribuicao"] < 0]
    frases = [f"O modelo estima uma probabilidade de DRC de {_pct(prob)}."]
    if escala == "probabilidade":
        frases[0] += f" Sem olhar para este doente, partiria de {_pct(base)}."
    if sobe:
        frases.append(f"O que mais aumentou o risco: {lista(sobe)}.")
    if desce:
        frases.append(f"O que mais o diminuiu: {lista(desce)}.")
    if escala != "probabilidade":
        frases.append("(Fatores por ordem de peso na decisão do modelo.)")
    if em_falta:
        frases.append("Em falta e substituídos por valores típicos do treino: "
                      + ", ".join(nome(v) for v in em_falta) + ".")
    return " ".join(frases)


@lru_cache(maxsize=1)
def obter_explicador():
    """Explicador pronto a usar (o agente e a app usam este)."""
    X_tr, _, _, _ = dados_divididos()
    return Explicador(carregar_modelo(), X_tr)


# ------------------------------------------------------------------
# Script: figuras e tabelas para o paper
# ------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, X_te, _, y_te = dados_divididos()
    final = carregar_modelo()
    expl = obter_explicador()
    valores, base = expl.calcular(X_te)
    escala = T["escala"][expl.escala]

    erro = float(np.max(np.abs(base + valores.sum(axis=1) - expl.saida_modelo(X_te))))
    print(f"Modelo: {final['nome']} | limiar {final['limiar']:.3f} | escala SHAP: {expl.escala}")
    print(f"Fidelidade: erro máximo |base + soma(SHAP) - saída do modelo| = {erro:.2e}")

    # 1. Importância global
    importancia = pd.Series(np.abs(valores).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    top = importancia.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([nome(v, LANG) for v in top.index], top.values,
            color=["#C0392B" if v in MARCADORES_KDIGO else "#2E86C1" for v in top.index])
    nome_modelo = final["nome"] if LANG == "pt" else NOMES_EN.get(final["nome"], final["nome"])
    ax.set(xlabel=T["imp"].format(e=escala), title=T["imp_t"].format(m=nome_modelo))
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_importancia_global.png", dpi=150)
    plt.close(fig)

    # 2. Dispersão por doente (estilo beeswarm)
    usados = pd.DataFrame(expl.imputador.transform(X_te), columns=FEATURES)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pos, var in enumerate(top.index):
        v = usados[var].to_numpy(dtype=float)
        cor = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.full_like(v, 0.5)
        pontos = ax.scatter(valores[:, FEATURES.index(var)], pos + rng.uniform(-0.25, 0.25, len(v)),
                            c=cor, cmap="coolwarm", s=14, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([nome(v, LANG) for v in top.index])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlabel=T["disp_x"].format(e=escala), title=T["disp_t"])
    fig.colorbar(pontos, ax=ax, label=T["cor"])
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_dispersao_global.png", dpi=150)
    plt.close(fig)

    # 3. Texto para cada doente do teste
    linhas = []
    for idx in X_te.index:
        e = expl.explicar_doente(X_te.loc[[idx]])
        linhas.append({"id": idx, "real": "DRC" if y_te[idx] else "sem DRC",
                       "probabilidade": e["probabilidade_drc"],
                       "decisao_modelo": "DRC" if e["probabilidade_drc"] >= final["limiar"] else "sem DRC",
                       "top5": ", ".join(f["variavel"] for f in e["fatores_principais"]),
                       "texto": e["texto"]})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(TABELAS / "shap_explicacoes_teste.csv", index=False, encoding="utf-8-sig")

    # 4. Três doentes de exemplo
    exemplos = {
        "drc": tabela.loc[tabela["probabilidade"].idxmax(), "id"],
        "sem_drc": tabela.loc[tabela["probabilidade"].idxmin(), "id"],
        "incerto": tabela.loc[(tabela["probabilidade"] - final["limiar"]).abs().idxmin(), "id"],
    }
    for rotulo, idx in exemplos.items():
        e = expl.explicar_doente(X_te.loc[[idx]], n_fatores=10)
        fs = e["fatores_principais"][::-1]
        etiquetas = [f["descricao"] if LANG == "pt" else
                     f"{nome(f['variavel'], 'en')} = {f['valor_usado']:g}" for f in fs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(etiquetas, [f["contribuicao"] for f in fs],
                color=["#C0392B" if f["contribuicao"] > 0 else "#2E86C1" for f in fs])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set(xlabel=T["doente_x"].format(e=escala),
               title=T["doente_t"].format(i=idx, r=T["rotulos"][rotulo], p=e["probabilidade_drc"]))
        fig.tight_layout()
        fig.savefig(FIGURAS / f"shap_doente_{idx}.png", dpi=150)
        plt.close(fig)
        print(f"\nDoente {idx} ({rotulo}): {e['texto']}")

    # 5. Avaliação da explicabilidade (proposta): marcadores KDIGO no top 5
    top5 = list(importancia.head(5).index)
    pd.DataFrame([{
        "modelo": final["nome"], "top5_shap": ", ".join(top5),
        "marcadores_kdigo_no_top5": ", ".join(v for v in top5 if v in MARCADORES_KDIGO),
        "n_marcadores_no_top5": sum(v in MARCADORES_KDIGO for v in top5),
        "erro_aditividade": erro,
    }]).to_csv(TABELAS / "shap_top5_kdigo.csv", index=False, encoding="utf-8-sig")
    print(f"\nTop 5 SHAP: {top5} -> {sum(v in MARCADORES_KDIGO for v in top5)} de 5 são marcadores KDIGO")
    print("Figuras em results/figures/ e tabelas em results/tables/")


if __name__ == "__main__":
    main()
