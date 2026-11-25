"""
  1. apaga valores impossiveis (passam a NaN = "em falta")
  2. converte as colunas de texto em 0/1
  3. separa as variaveis (X) do diagnostico (y)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERICAS = [
    "age",    # idade
    "bp",     # tensão arterial
    "sg",     # densidade da urina
    "al",     # albumina na urina
    "su",     # açúcar na urina
    "bgr",    # glicose aleatória
    "bu",     # ureia
    "sc",     # creatinina
    "sod",    # sódio
    "pot",    # potássio
    "hemo",   # hemoglobina
    "pcv",    # hematócrito
    "wbcc",   # leucócitos
    "rbcc",   # eritrócitos
]
 
# Conversão binária -> numero. 1 = o valor "mais doente", 0 = o valor "normal".
CATEGORICAS = {
    "rbc":   {"normal": 0, "abnormal": 1},      # glóbulos vermelhos na urina
    "pc":    {"normal": 0, "abnormal": 1},      # células de pus
    "pcc":   {"notpresent": 0, "present": 1},   # aglomerados de pus
    "ba":    {"notpresent": 0, "present": 1},   # bactérias
    "htn":   {"no": 0, "yes": 1},               # hipertensão
    "dm":    {"no": 0, "yes": 1},               # diabetes
    "cad":   {"no": 0, "yes": 1},               # doença coronária
    "appet": {"good": 0, "poor": 1},            # apetite
    "pe":    {"no": 0, "yes": 1},               # edema
    "ane":   {"no": 0, "yes": 1},               # anemia
}

ALVO = "class"
DOENCA = "ckd"   # 1 = ckd, 0 = notckd

# Fora destes limites o valor é incompatível com a vida -> erro de registo.
LIMITES = {
    "age":  (0, 120),         # anos
    "bp":   (20, 300),        # mmHg
    "sg":   (1.000, 1.060),   # adimensional
    # "al":  escala ordinal (0-5), sem limite fisico
    # "su":  escala ordinal (0-5), sem limite fisico
    "bgr":  (20, 1500),       # mg/dL
    "bu":   (5, None),        # mg/dL; encontrado 1,5 (com creatinina 7,3)
    "sc":   (0.1, 80),        # mg/dL
    "sod":  (100, 180),       # mEq/L; encontrado 4,5
    "pot":  (1.5, 10),        # mEq/L; encontrados 39 e 47
    "hemo": (2, 25),          # g/dL
    "pcv":  (5, 75),          # %
    "wbcc": (100, 200000),    # cells/cumm
    "rbcc": (1.0, 9.0),       # milhoes/cumm
}

def apagar_impossiveis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Valores fora dos limites de sobrevivência
    for col, (lo, hi) in LIMITES.items():
        impossivel = pd.Series(False, index=df.index)
        if lo is not None:
            impossivel |= df[col] < lo
        if hi is not None:
            impossivel |= df[col] > hi
        df.loc[impossivel, col] = np.nan

    # 2) Incoerência específica: creatinina 48,1 mg/dL com ureia normal (26).
    #    Tratada caso a caso: uma regra geral creatinina/ureia seria pouco
    #    fiável (diálise e dieta pobre em proteína baixam a ureia).
    incoerente = (df["sc"] == 48.1) & (df["bu"] == 26.0)
    df.loc[incoerente, "sc"] = np.nan

    return df


def preparar(caminho: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(caminho)
    df = apagar_impossiveis(df)

    for col, mapa in CATEGORICAS.items():
        df[col] = df[col].map(mapa)   # vazios continuam NaN
    y = (df[ALVO] == DOENCA).astype(int)   # 1 = CKD, 0 = sem CKD
    X = df[NUMERICAS + list(CATEGORICAS)]
    return X, y


SEED = 42


def dividir(X, y, seed: int = SEED):
    """Divisão 80/20 estratificada, igual para todo o projeto.
    Devolve X_tr, X_te, y_tr, y_te. Usar SEMPRE esta função (SHAP, agente, app)
    para treino e teste serem os mesmos."""
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


if __name__ == "__main__":
    import sys
    X, y = preparar(sys.argv[1] if len(sys.argv) > 1 else "ckd_uci.csv")
    print(f"X: {X.shape[0]} doentes x {X.shape[1]} variáveis")
    print(f"y: {y.sum()} com CKD, {(y == 0).sum()} sem CKD")
    print(f"Tipos de dados em X: {X.dtypes.value_counts().to_dict()}")
    print("\nPrimeiros 3 doentes:")
    print(X.head(3).to_string())