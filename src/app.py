"""
app.py
Interface Streamlit do sistema de apoio à decisão para DRC.

Requer ter corrido antes: comparar_modelos.py e limiar_calibracao.py.
Uso (a partir da raiz do projeto):
    streamlit run src/app.py

  1. Avaliação do doente: o médico preenche os valores e vê risco, decisão do modelo,
     explicação SHAP e recomendação de referenciação (funciona sem chave da API).
  2. Conversar com o agente (Claude + ferramentas), se ANTHROPIC_API_KEY estiver definida.
  3. Casos a confirmar: active learning, os casos mais incertos primeiro.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from descricoes import CATEGORICAS, DESCRICOES, FEATURES, nome
from explicar_shap import carregar_modelo, dados_divididos
from agente import Agente, chave_disponivel
from ferramentas import (ESSENCIAIS, ZONA_INCERTA, avaliar_referenciacao, explicar,
                         fila_para_confirmar, prever_risco, registar_confirmacao)

st.set_page_config(page_title="Apoio à decisão DRC", page_icon="🩺", layout="wide")
OPCOES_CAT = ["(em falta)", "não", "sim"]

final = carregar_modelo()
_, X_te, _, y_te = dados_divididos()


def chave(v):
    return f"campo_{v}"

def pct(p):
    """Mostra a probabilidade sem prometer certeza: o modelo não está calibrado."""
    if p > 0.99:
        return ">99%"
    if p < 0.01:
        return "<1%"
    return f"{p:.0%}"

for v in FEATURES:
    st.session_state.setdefault(chave(v), "(em falta)" if v in CATEGORICAS else None)


def carregar_exemplo():
    idx = st.session_state["exemplo"]
    for v in FEATURES:
        valor = X_te.loc[idx, v]
        if v in CATEGORICAS:
            st.session_state[chave(v)] = "(em falta)" if pd.isna(valor) else OPCOES_CAT[int(valor) + 1]
        else:
            st.session_state[chave(v)] = None if pd.isna(valor) else float(valor)
    st.session_state["doente_id"] = f"teste-{idx}"


def limpar_formulario():
    for v in FEATURES:
        st.session_state[chave(v)] = "(em falta)" if v in CATEGORICAS else None
    st.session_state.pop("avaliacao", None)

def novo_doente():
    """Começa de novo: limpa formulário, identificador, avaliação e conversa com o agente."""
    limpar_formulario()
    st.session_state["doente_id"] = ""
    st.session_state.pop("agente", None)
    st.session_state["mensagens"] = []

def confirmar(diagnostico):
    av = st.session_state["avaliacao"]
    registar_confirmacao(st.session_state.get("doente_id") or None, av["dados"],
                         av["risco"]["probabilidade_drc"], av["risco"]["decisao_modelo"],
                         diagnostico)
    av["confirmado"] = diagnostico

def confirmar_da_fila(caso, diagnostico):
    p = caso["probabilidade_drc"]
    registar_confirmacao(caso["doente_id"], caso["dados"], p,
                         "DRC" if p >= final["limiar"] else "sem DRC", diagnostico)

def dados_do_formulario():
    dados = {}
    for v in FEATURES:
        valor = st.session_state[chave(v)]
        if v in CATEGORICAS:
            if valor != "(em falta)":
                dados[v] = 1 if valor == "sim" else 0
        elif valor is not None:
            dados[v] = float(valor)
    return dados


# ------------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Modelo")
    st.write(f"**{final['nome']}**")
    st.caption(f"Limiar de decisão: {final['limiar']:.2f} (sensibilidade-alvo do grupo) · "
               f"probabilidade: {final.get('probabilidade', 'original')} (escolhida pelo Brier)")
    st.caption(f"Zona de incerteza: {ZONA_INCERTA[0]:.0%} a {ZONA_INCERTA[1]:.0%} "
               "(casos deferidos ao médico)")
    st.header("Doente de exemplo")
    st.selectbox("Doente do conjunto de teste", list(X_te.index), key="exemplo",
                 format_func=lambda i: f"#{i} (real: {'DRC' if y_te[i] else 'sem DRC'})")
    st.button("Carregar valores", on_click=carregar_exemplo, use_container_width=True)
    st.button("Limpar formulário", on_click=limpar_formulario, use_container_width=True)
    st.caption("O diagnóstico real só aparece aqui para demonstração.")

st.title("Apoio à decisão: deteção precoce de doença renal crónica")
st.warning("**Este sistema não substitui o médico.** É um protótipo académico de apoio à "
           "decisão (HCAI 2026/27), não um dispositivo médico: o diagnóstico e a decisão de "
           "referenciação são sempre do médico.", icon="⚠️")
aba_avaliacao, aba_fila, aba_chat = st.tabs(
    ["Avaliação do doente", "Casos a confirmar", "Conversar com o agente"])

# ------------------------------------------------------------------
# 1. Avaliação do doente
# ------------------------------------------------------------------
with aba_avaliacao:
    st.button("Novo doente", on_click=novo_doente, key="novo_doente_form",
              help="Limpa o formulário, o identificador e a avaliação.")
    st.text_input("Identificador do doente (opcional, para comparar consultas)", key="doente_id")
    st.markdown("**Valores do doente** (deixe em branco o que não tem; * = essencial)")
    colunas = st.columns(4)
    for i, v in enumerate(FEATURES):
        info = DESCRICOES[v]
        rotulo = info["nome"] + (f" ({info['unidade']})" if info["unidade"] else "")
        rotulo += " *" if v in ESSENCIAIS else ""
        with colunas[i % 4]:
            if v in CATEGORICAS:
                st.selectbox(rotulo, OPCOES_CAT, key=chave(v))
            else:
                st.number_input(rotulo, key=chave(v), format="%g", placeholder="em falta")

    if st.button("Avaliar", type="primary"):
        dados = dados_do_formulario()
        try:
            doente_id = st.session_state.get("doente_id") or None
            st.session_state["avaliacao"] = {
                "dados": dados,
                "risco": prever_risco(dados, doente_id=doente_id),
                "explicacao": explicar(dados, n_fatores=8),
                "referenciacao": avaliar_referenciacao(dados),
            }
        except ValueError as erro:   # ex.: valor fora dos limites possíveis
            st.session_state.pop("avaliacao", None)
            st.error(str(erro))

    av = st.session_state.get("avaliacao")
    if av:
        risco, expl, ref = av["risco"], av["explicacao"], av["referenciacao"]
        st.divider()
        for aviso_sc in risco.get("avisos_sc", []):
            st.warning(aviso_sc)
        if risco["essenciais_em_falta"]:
            st.info("Faltam valores essenciais: "
                    + ", ".join(e["nome"] for e in risco["essenciais_em_falta"])
                    + ". Sem eles não se mostra probabilidade, decisão, explicação nem referenciação.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Probabilidade de DRC", pct(risco["probabilidade_drc"]))        
            c2.metric("Decisão do modelo", risco["decisao_modelo"],
                    help=f"DRC se a probabilidade for >= {risco['limiar']:.2f}")
            c3.metric("Categoria", risco["categoria"])
            comp = risco.get("comparacao_consulta_anterior")
            if comp:
                var = round(comp["variacao_probabilidade"] * 100)
                c4.metric("Vs consulta anterior", "sem alteração" if var == 0 else f"{var:+d} p.p.")
            if risco["deferir_ao_medico"]:
                st.warning("O modelo não tem certeza neste caso: deve ser avaliado pelo médico, "
                        "com confirmação por exames.")
            if comp and comp["valores_alterados"]:
                st.caption("Alterações desde a consulta anterior: " + "; ".join(comp["valores_alterados"]))

            st.subheader("Porque é que o modelo deu este resultado?")
            fs = expl["fatores_principais"][::-1]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh([f["descricao"] for f in fs], [f["contribuicao"] for f in fs],
                    color=["#C0392B" if f["contribuicao"] > 0 else "#2E86C1" for f in fs])
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel(f"Contribuição SHAP ({expl['escala']}); vermelho aumenta o risco")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.write(expl["texto"])
            st.caption(expl["aviso"])

            st.subheader("Referenciação")
            st.write(f"**{ref['recomendacao']}**")
            if risco["decisao_modelo"] == "sem DRC" and ref["criterios_cumpridos"]:
                st.warning("O modelo indica sem DRC, mas há critérios de referenciação cumpridos. "
                        "O modelo avalia o padrão global e os critérios avaliam valores isolados, "
                        "por isso podem divergir: avaliar clinicamente.")
            for c in ref["criterios_cumpridos"]:
                st.write(f"- {c}")
            if ref["criterios_nao_avaliaveis_por_falta_de_dados"]:
                st.caption("Critérios não avaliáveis (faltam dados): "
                        + ", ".join(ref["criterios_nao_avaliaveis_por_falta_de_dados"]))
            st.caption(ref["aviso"])
            if not risco["essenciais_em_falta"]:
                st.subheader("Confirmação do médico")
                if "confirmado" in av:
                    st.success("Confirmação registada: " + ("DRC" if av["confirmado"] else "sem DRC") + ".")
                else:
                    st.caption("Depois dos exames, regista o diagnóstico. Serve para a aprendizagem ativa.")
                    b1, b2 = st.columns(2)
                    b1.button("Confirmo: DRC", on_click=confirmar, args=(1,))
                    b2.button("Confirmo: sem DRC", on_click=confirmar, args=(0,))
            st.info("Lembrete: o resultado acima é uma sugestão do modelo. Confirme com a avaliação "
                    "clínica e os exames antes de qualquer decisão.")

# ------------------------------------------------------------------
# Casos a confirmar (active learning: os mais incertos primeiro)
# ------------------------------------------------------------------
with aba_fila:
    st.markdown("Consultas avaliadas e ainda sem confirmação, **das mais incertas para as menos "
                "incertas**: são as que mais ajudam o modelo a aprender.")
    fila = fila_para_confirmar()
    if not fila:
        st.info("Não há casos por confirmar. Avalie doentes com identificador na primeira aba.")
    else:
        st.dataframe(pd.DataFrame([{"Doente": c["doente_id"],
                                    "Probabilidade de DRC": pct(c["probabilidade_drc"]),
                                    "Incerteza (0 a 0,5)": c["incerteza"],
                                    "Data": c["data"]} for c in fila]), hide_index=True)
        i = st.selectbox("Caso a confirmar", range(len(fila)),
                         format_func=lambda k: f"{fila[k]['doente_id']} ({pct(fila[k]['probabilidade_drc'])})")
        b1, b2 = st.columns(2)
        b1.button("Confirmo: DRC", on_click=confirmar_da_fila, args=(fila[i], 1), key="fila_drc")
        b2.button("Confirmo: sem DRC", on_click=confirmar_da_fila, args=(fila[i], 0), key="fila_nao")

# ------------------------------------------------------------------
# 2. Chat com o agente
# ------------------------------------------------------------------
with aba_chat:
    if not chave_disponivel():
        st.info("Para conversar com o agente, defina OPENROUTER_API_KEY (ou ANTHROPIC_API_KEY) antes de correr "
                "`streamlit run src/app.py` (ver README). A avaliação do doente funciona sem ela.")
    else:
        if "agente" not in st.session_state:
            st.session_state["agente"] = Agente()
            st.session_state["mensagens"] = []
        st.button("Novo doente", on_click=novo_doente,
                  help="Limpa o formulário, o identificador e a conversa com o agente.")
        incluir = st.checkbox("Enviar também os valores preenchidos no formulário", value=True)
        for m in st.session_state["mensagens"]:
            with st.chat_message(m["papel"]):
                st.markdown(m["texto"])
                for c in m.get("chamadas", []):
                    with st.expander(f"Ferramenta usada: {c['ferramenta']}"):
                        st.json(json.loads(c["saida"]))

        pergunta = st.chat_input("Ex.: Doente de 64 anos, hipertenso, creatinina 2.3...")
        if pergunta:
            texto, dados = pergunta, dados_do_formulario()
            if incluir and dados:
                texto += f"\n\n[Valores do formulário: {json.dumps(dados, ensure_ascii=False)}]"
                if st.session_state.get("doente_id"):
                    texto += f"\n[Identificador do doente: {st.session_state['doente_id']}]"
            st.session_state["mensagens"].append({"papel": "user", "texto": pergunta})
            with st.spinner("O agente está a analisar..."):
                resposta, chamadas = st.session_state["agente"].perguntar(texto)
            st.session_state["mensagens"].append({"papel": "assistant", "texto": resposta,
                                                  "chamadas": chamadas})
            st.rerun()
