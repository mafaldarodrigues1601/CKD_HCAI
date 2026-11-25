"""
ferramentas.py
Ferramentas que o agente (LLM) pode chamar. O LLM só gere o diálogo; o risco,
a decisão (limiar do grupo), a incerteza e as explicações vêm SEMPRE daqui.

  prever_risco           probabilidade de DRC, decisão do modelo com o limiar do
                         limiar_calibracao.py, categoria (baixo / incerto / elevado),
                         valores essenciais em falta e comparação com a consulta anterior
  explicar               fatores SHAP principais deste doente, em linguagem simples
  avaliar_referenciacao  critérios simplificados (KDIGO 2024) para referenciar à nefrologia
  consultar_historico    consultas anteriores do mesmo doente

Também se usam diretamente em Python (a app usa-as assim):
    from ferramentas import prever_risco
    prever_risco({"age": 60, "sc": 2.1, "hemo": 10.2, "htn": "sim"})
"""
import json
from datetime import datetime

import numpy as np

from config import TABELAS
from descricoes import CATEGORICAS, DESCRICOES, FEATURES, doente_para_dataframe, nome
from explicar_shap import carregar_modelo, obter_explicador, _pct

# Entre 0.25 e 0.75 o modelo não tem certeza suficiente: o caso é deferido ao médico
# (Human-in-the-Loop). O limiar de decisão do grupo fica dentro desta zona.
ZONA_INCERTA = (0.25, 0.75)

# Valores de rotina sem os quais o agente pede dados antes de concluir.
ESSENCIAIS = ["age", "bp", "sg", "al", "sc", "bu", "hemo", "htn", "dm"]

CAMINHO_CONSULTAS = TABELAS.parent / "consultas.json"
CAMINHO_ROTULOS = TABELAS.parent / "rotulos_confirmados.json"
AVISO = "Apoio à decisão: não é um diagnóstico. A decisão final é sempre do médico."


# ------------------------------------------------------------------
# Histórico de consultas
# ------------------------------------------------------------------
def _ler_consultas():
    if CAMINHO_CONSULTAS.exists():
        with open(CAMINHO_CONSULTAS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _guardar_consulta(doente_id, dados, probabilidade, categoria, provisorio=False):
    consultas = _ler_consultas()
    consultas.setdefault(str(doente_id), []).append({
        "data": datetime.now().isoformat(timespec="seconds"),
        "dados": dados, "probabilidade_drc": probabilidade, "categoria": categoria,
        "provisorio": provisorio})
    with open(CAMINHO_CONSULTAS, "w", encoding="utf-8") as f:
        json.dump(consultas, f, indent=2, ensure_ascii=False)


def _comparar(anterior, dados, probabilidade):
    mudancas = [f"{nome(v)}: {anterior['dados'][v]} -> {dados[v]}" for v in FEATURES
                if v in anterior["dados"] and v in dados and anterior["dados"][v] != dados[v]]
    return {"data_anterior": anterior["data"],
            "probabilidade_anterior": anterior["probabilidade_drc"],
            "variacao_probabilidade": round(probabilidade - anterior["probabilidade_drc"], 4),
            "valores_alterados": mudancas}


def _limpar(dados):
    return {k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in (dados or {}).items() if v is not None and v != ""}

LIMITE_AVISO_CREATININA = 20  # mg/dL; escolha de segurança, não vem de nenhuma guideline

def _avisos_unidades(dados):
    avisos = []
    sc = dados.get("sc")
    if sc is not None and sc > LIMITE_AVISO_CREATININA:
        avisos.append(
            f"Creatinina invulgarmente alta ({sc} mg/dL): confirma a unidade "
            "(mg/dL; 180 µmol/L correspondem a cerca de 2,0 mg/dL)."
        )
    return avisos

# ------------------------------------------------------------------
# Ferramentas
# ------------------------------------------------------------------
def prever_risco(dados, doente_id=None):
    dados = _limpar(dados)
    final = carregar_modelo()
    prob = float(final["modelo"].predict_proba(doente_para_dataframe(dados))[0, 1])
    limiar = float(final["limiar"])

    if prob < ZONA_INCERTA[0]:
        categoria = "baixo"
    elif prob > ZONA_INCERTA[1]:
        categoria = "elevado"
    else:
        categoria = "incerto"
    essenciais_em_falta = [v for v in ESSENCIAIS if v not in dados]

    resultado = {
        "probabilidade_drc": round(prob, 4),
        "decisao_modelo": "DRC" if prob >= limiar else "sem DRC",
        "limiar": round(limiar, 4),
        "categoria": categoria,
        "deferir_ao_medico": categoria == "incerto",
        "modelo": final["nome"],
        # "original" ou "calibrado (sigmoide)": escolhido pelo Brier no limiar_calibracao.py
        "calibracao": final.get("probabilidade", "original"),
        "variaveis_em_falta": [v for v in FEATURES if v not in dados],
        "essenciais_em_falta": [{"variavel": v, "nome": nome(v), "unidade": DESCRICOES[v]["unidade"]}
                                for v in essenciais_em_falta],
        "resultado_provisorio": bool(essenciais_em_falta),
        "aviso": AVISO,
    }

    resultado["probabilidade_texto"] = _pct(resultado["probabilidade_drc"])

    avisos_sc = _avisos_unidades(dados)
    if avisos_sc:
        resultado["avisos_sc"] = avisos_sc

    if doente_id is not None:
        # só se compara com consultas completas, e só se esta também for completa
        anteriores = [c for c in _ler_consultas().get(str(doente_id), []) if not c.get("provisorio")]
        if anteriores and not essenciais_em_falta:
            resultado["comparacao_consulta_anterior"] = _comparar(anteriores[-1], dados, prob)
        _guardar_consulta(doente_id, dados, round(prob, 4), categoria,
                          provisorio=bool(essenciais_em_falta))
    return resultado


def explicar(dados, n_fatores=5):
    e = obter_explicador().explicar_doente(doente_para_dataframe(_limpar(dados)), n_fatores)
    e["aviso"] = "O SHAP mostra o que pesou na decisão do MODELO, não causas clínicas."
    return e


def avaliar_referenciacao(dados):
    """Critérios simplificados, inspirados na KDIGO 2024, para discussão com o médico.
    Sem sexo no conjunto de dados não se calcula a TFGe: usam-se aproximações.
    Os limiares devem ser validados por um clínico (uso académico)."""
    dados = _limpar(dados)
    X = doente_para_dataframe(dados).iloc[0]
    risco = prever_risco(dados)
    criterios = []

    if not np.isnan(X["al"]) and X["al"] >= 3:
        criterios.append("Albumina na urina >= 3 na tira: sugere proteinúria marcada "
                         "(aprox. categoria A3); confirmar com RAC.")
    if not np.isnan(X["sc"]) and X["sc"] >= 2.0:
        criterios.append("Creatinina >= 2.0 mg/dL: aproximação a TFGe < 30 mL/min/1,73 m² em adultos; "
                         "calcular a TFGe com idade e sexo.")
    if not np.isnan(X["pot"]) and X["pot"] > 5.5:
        criterios.append("Potássio > 5.5 mEq/L: alteração persistente do potássio é motivo de "
                         "referenciação (> 6.0 pode exigir avaliação urgente).")
    if not np.isnan(X["rbc"]) and X["rbc"] == 1 and not np.isnan(X["al"]) and X["al"] >= 1:
        criterios.append("Eritrócitos na urina anormais com albumina: hematúria a esclarecer.")

    if criterios:
        recomendacao = "Considerar referenciação para nefrologia."
    elif risco["categoria"] == "incerto":
        recomendacao = "O modelo não tem certeza: o médico deve avaliar e confirmar com exames."
    elif risco["decisao_modelo"] == "DRC":
        recomendacao = ("Risco elevado sem critérios de referenciação: confirmar DRC nos cuidados "
                        "primários (TFGe e RAC) e repetir em 3 meses (a DRC exige alterações "
                        "persistentes há mais de 3 meses).")
    else:
        recomendacao = ("Sem indicação de referenciação. Manter rastreio periódico se tiver "
                        "hipertensão ou diabetes.")

    return {
        "probabilidade_drc": risco["probabilidade_drc"],
        "decisao_modelo": risco["decisao_modelo"],
        "criterios_cumpridos": criterios,
        "recomendacao": recomendacao,
        "criterios_nao_avaliaveis_por_falta_de_dados":
            [nome(v) for v in ("al", "sc", "pot", "rbc") if v not in dados],
        "aviso": AVISO + " Critérios simplificados para fins académicos.",
    }


def consultar_historico(doente_id):
    consultas = _ler_consultas().get(str(doente_id), [])
    return {"doente_id": str(doente_id), "numero_consultas": len(consultas),
            "consultas": consultas[-5:]}

def _ler_rotulos():
    try:
        with open(CAMINHO_ROTULOS, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def registar_confirmacao(doente_id, dados, probabilidade, decisao_modelo, diagnostico_medico):
    """Guarda o diagnóstico confirmado pelo médico (1 = DRC, 0 = sem DRC).
    Não é uma ferramenta do agente: só a interface a chama, depois de o médico confirmar."""
    if diagnostico_medico not in (0, 1):
        raise ValueError("diagnostico_medico tem de ser 0 (sem DRC) ou 1 (DRC)")
    if any(v not in dados for v in ESSENCIAIS):
        raise ValueError("faltam valores essenciais: não se regista confirmação")
    rotulos = _ler_rotulos()
    agora = datetime.now().isoformat(timespec="seconds")
    rotulos.append({
        "doente_id": str(doente_id) if doente_id else f"sem-id-{agora}",
        "data": agora,
        "dados": dados,
        "probabilidade_drc": probabilidade,
        "decisao_modelo": decisao_modelo,
        "diagnostico_medico": diagnostico_medico,
        "modelo_acertou": (decisao_modelo == "DRC") == bool(diagnostico_medico),
    })
    CAMINHO_ROTULOS.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMINHO_ROTULOS, "w", encoding="utf-8") as f:
        json.dump(rotulos, f, indent=2, ensure_ascii=False)

def fila_para_confirmar():
    """Consultas completas ainda sem confirmação do médico, das mais incertas para as
    menos incertas. Incerteza = 1 - max(p, 1-p) (least confidence, como no active_learning_ckd.py)."""
    confirmados = {r["doente_id"] for r in _ler_rotulos()}
    fila = []
    for doente_id, consultas in _ler_consultas().items():
        completas = [c for c in consultas if not c.get("provisorio")]
        if not completas or doente_id in confirmados:
            continue
        ultima = completas[-1]
        p = ultima["probabilidade_drc"]
        fila.append({"doente_id": doente_id, "data": ultima["data"], "dados": ultima["dados"],
                     "probabilidade_drc": p, "incerteza": round(1 - max(p, 1 - p), 4)})
    return sorted(fila, key=lambda c: -c["incerteza"])

# ------------------------------------------------------------------
# Definição das ferramentas para a API do Claude (tool use)
# ------------------------------------------------------------------
def _esquema_dados():
    props = {}
    for v in FEATURES:
        info = DESCRICOES[v]
        if v in CATEGORICAS:
            props[v] = {"type": "integer", "enum": [0, 1],
                        "description": f"{info['nome']} (1 = sim/presente/anormal, 0 = não)"}
        else:
            unidade = f" em {info['unidade']}" if info["unidade"] else ""
            props[v] = {"type": "number", "description": f"{info['nome']}{unidade}"}
    return {"type": "object", "properties": props, "additionalProperties": False,
            "description": "Valores do doente. Só os que o médico deu; nunca inventar os que faltam."}


FERRAMENTAS = [
    {"name": "prever_risco",
     "description": ("Probabilidade de doença renal crónica (DRC) com o modelo do projeto, decisão "
                     "com o limiar escolhido, categoria (baixo/incerto/elevado), se o caso deve ser "
                     "deferido ao médico, valores essenciais em falta e, com doente_id, comparação "
                     "com a consulta anterior. Usar sempre antes de falar de risco."),
     "input_schema": {"type": "object",
                      "properties": {"dados": _esquema_dados(),
                                     "doente_id": {"type": "string",
                                                   "description": "Identificador do doente, se dado."}},
                      "required": ["dados"]}},
    {"name": "explicar",
     "description": "Explica com SHAP as variáveis que mais aumentaram ou diminuíram o risco deste doente.",
     "input_schema": {"type": "object", "properties": {"dados": _esquema_dados()},
                      "required": ["dados"]}},
    {"name": "avaliar_referenciacao",
     "description": ("Critérios clínicos simplificados (KDIGO 2024) e recomendação de "
                     "referenciação para nefrologia."),
     "input_schema": {"type": "object", "properties": {"dados": _esquema_dados()},
                      "required": ["dados"]}},
    {"name": "consultar_historico",
     "description": "Consultas anteriores registadas para um doente.",
     "input_schema": {"type": "object", "properties": {"doente_id": {"type": "string"}},
                      "required": ["doente_id"]}},
]

_FUNCOES = {"prever_risco": prever_risco, "explicar": explicar,
            "avaliar_referenciacao": avaliar_referenciacao,
            "consultar_historico": consultar_historico}


def _bloquear_se_incompleto(nome_ferramenta, argumentos, resultado):
    """Garantia no código (não só no prompt): sem os valores essenciais, o agente não
    vê probabilidade, decisão, explicação nem referenciação. A app chama as funções
    diretamente e continua a mostrar o resultado provisório."""
    if nome_ferramenta not in ("prever_risco", "explicar", "avaliar_referenciacao"):
        return resultado
    dados = _limpar(argumentos.get("dados"))
    falta = [v for v in ESSENCIAIS if v not in dados]
    if not falta:
        return resultado
    pedir = [{"variavel": v, "nome": nome(v), "unidade": DESCRICOES[v]["unidade"]} for v in falta]
    if nome_ferramenta == "prever_risco":
        return {"resultado_ocultado": "Faltam valores essenciais: não há probabilidade nem decisão. "
                                      "Pedir ao médico os valores em essenciais_em_falta.",
                "essenciais_em_falta": pedir, "resultado_provisorio": True, "aviso": AVISO}
    return {"erro_dados_insuficientes": "Faltam valores essenciais; pedir ao médico.",
            "essenciais_em_falta": pedir}


def executar_ferramenta(nome_ferramenta, argumentos):
    """Corre uma ferramenta e devolve JSON (texto) para o LLM. Erros voltam ao LLM,
    que os explica ao médico (ex.: valor fora dos limites possíveis)."""
    try:
        resultado = _FUNCOES[nome_ferramenta](**argumentos)
        resultado = _bloquear_se_incompleto(nome_ferramenta, argumentos, resultado)
    except Exception as erro:
        resultado = {"erro": f"{type(erro).__name__}: {erro}"}
    return json.dumps(resultado, ensure_ascii=False)