"""
descricoes.py
Nomes legíveis, unidades e intervalos de referência das 24 variáveis, e conversão
dos dados de UM doente (escritos pelo médico) para o formato do modelo.

Usado pelo SHAP (explicar_shap.py), pelas ferramentas do agente (ferramentas.py)
e pela app (app.py). Não altera o Prep_Data.py: só o usa.

Os intervalos de referência são aproximados (adultos) e servem só para escrever
"abaixo/acima do normal" nas explicações; devem ser confirmados com um clínico.
"""
import numpy as np
import pandas as pd

from Prep_Data import CATEGORICAS, LIMITES, NUMERICAS

FEATURES = NUMERICAS + list(CATEGORICAS)   # mesma ordem do preparar()

# variável: (nome PT, nome EN, unidade, intervalo de referência ou None)
_INFO = {
    "age":  ("idade", "age", "anos", None),
    "bp":   ("pressão arterial diastólica", "diastolic blood pressure", "mmHg", (60, 80)),
    "sg":   ("densidade urinária", "urine specific gravity", "", (1.015, 1.025)),
    "al":   ("albumina na urina (0-5)", "urine albumin (0-5)", "", (0, 0)),
    "su":   ("açúcar na urina (0-5)", "urine sugar (0-5)", "", (0, 0)),
    "bgr":  ("glicemia ocasional", "random blood glucose", "mg/dL", (70, 140)),
    "bu":   ("ureia", "blood urea", "mg/dL", (15, 45)),
    "sc":   ("creatinina", "serum creatinine", "mg/dL", (0.6, 1.2)),
    "sod":  ("sódio", "sodium", "mEq/L", (135, 145)),
    "pot":  ("potássio", "potassium", "mEq/L", (3.5, 5.0)),
    "hemo": ("hemoglobina", "haemoglobin", "g/dL", (12, 17)),
    "pcv":  ("hematócrito", "packed cell volume", "%", (36, 50)),
    "wbcc": ("leucócitos", "white blood cell count", "células/µL", (4000, 11000)),
    "rbcc": ("eritrócitos", "red blood cell count", "milhões/µL", (4.2, 6.1)),
    "rbc":  ("eritrócitos na urina anormais", "abnormal urine red blood cells", "", None),
    "pc":   ("células de pus na urina anormais", "abnormal pus cells", "", None),
    "pcc":  ("aglomerados de células de pus", "pus cell clumps", "", None),
    "ba":   ("bactérias na urina", "bacteria in urine", "", None),
    "htn":  ("hipertensão", "hypertension", "", None),
    "dm":   ("diabetes", "diabetes", "", None),
    "cad":  ("doença coronária", "coronary artery disease", "", None),
    "appet": ("falta de apetite", "poor appetite", "", None),
    "pe":   ("edema dos membros inferiores", "pedal oedema", "", None),
    "ane":  ("anemia", "anaemia", "", None),
}
DESCRICOES = {v: {"nome": pt, "nome_en": en, "unidade": u, "ref": ref}
              for v, (pt, en, u, ref) in _INFO.items()}
assert set(DESCRICOES) == set(FEATURES), "DESCRICOES tem de cobrir as 24 variáveis"


def nome(variavel, lang="pt"):
    return DESCRICOES[variavel]["nome" if lang == "pt" else "nome_en"]


def descrever_valor(variavel, valor):
    """Texto curto, ex.: 'hemoglobina 9.6 g/dL (abaixo do normal)'."""
    info = DESCRICOES[variavel]
    if pd.isna(valor):
        return f"{info['nome']}: em falta"
    if variavel in CATEGORICAS:
        return f"{info['nome']}: {'sim' if valor >= 0.5 else 'não'}"
    unidade = f" {info['unidade']}" if info["unidade"] else ""
    texto = f"{info['nome']} {valor:g}{unidade}"
    ref = info["ref"]
    if ref is not None:
        if valor < ref[0]:
            texto += " (abaixo do normal)"
        elif valor > ref[1]:
            texto += " (acima do normal)"
        else:
            texto += " (normal)"
    return texto


# ------------------------------------------------------------------
# Um doente individual
# ------------------------------------------------------------------
_TEXTO_PARA_NUM = {"sim": 1, "não": 0, "nao": 0, "presente": 1, "ausente": 0,
                   "anormal": 1, "normal": 0, "bom": 0, "mau": 1}
for _mapa in CATEGORICAS.values():
    _TEXTO_PARA_NUM.update(_mapa)


def doente_para_dataframe(dados):
    """Converte {variável: valor} numa linha com as 24 variáveis (NaN = em falta).

    Aceita 0/1 ou texto ("sim", "yes", "abnormal") nas categóricas.
    Dá erro (em vez de ignorar) para variáveis desconhecidas e para valores
    fora dos limites fisiológicos do Prep_Data.LIMITES, para o médico confirmar.
    """
    desconhecidas = set(dados) - set(FEATURES)
    if desconhecidas:
        raise ValueError(f"Variáveis desconhecidas: {sorted(desconhecidas)}. Válidas: {FEATURES}")

    linha = {}
    for col in FEATURES:
        valor = dados.get(col)
        if valor is None or (isinstance(valor, str) and not valor.strip()):
            linha[col] = np.nan
            continue
        if col in CATEGORICAS:
            if isinstance(valor, str):
                texto = valor.strip().lower()
                if texto not in _TEXTO_PARA_NUM:
                    raise ValueError(f"Valor inválido para {col}: {valor!r} (use sim/não ou 0/1)")
                valor = _TEXTO_PARA_NUM[texto]
            if float(valor) not in (0.0, 1.0):
                raise ValueError(f"{col} tem de ser 0 (não) ou 1 (sim), recebi {valor!r}")
            linha[col] = float(valor)
        else:
            valor = float(str(valor).replace(",", ".")) if isinstance(valor, str) else float(valor)
            lo, hi = LIMITES.get(col, (None, None))
            if col in ("al", "su") and not 0 <= valor <= 5:
                raise ValueError(f"{nome(col)} usa a escala 0-5, recebi {valor:g}")
            if (lo is not None and valor < lo) or (hi is not None and valor > hi):
                raise ValueError(f"{nome(col)} = {valor:g} {DESCRICOES[col]['unidade']} está fora "
                                 f"dos limites possíveis ({lo}-{hi}). Confirme o valor e a unidade.")
            linha[col] = valor
    return pd.DataFrame([linha], columns=FEATURES)
