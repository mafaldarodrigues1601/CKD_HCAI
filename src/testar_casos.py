"""
testar_casos.py
Testa o sistema completo (modelo final + limiar + SHAP + referenciação) em 3 casos
de exemplo do conjunto de teste, escolhidos automaticamente:
  * claramente DRC      -> doente com DRC com a maior probabilidade
  * caso limite         -> doente com a probabilidade mais perto do limiar
  * saudável            -> doente sem DRC com a menor probabilidade

Para cada caso corre as mesmas ferramentas que o agente usa (prever_risco, explicar,
avaliar_referenciacao) e verifica o comportamento esperado:
  * DRC      -> decisão "DRC" e categoria "elevado"
  * limite   -> deferido ao médico (zona de incerteza)
  * saudável -> decisão "sem DRC" e categoria "baixo"

Requer ter corrido antes: comparar_modelos.py e limiar_calibracao.py.
Uso (a partir da raiz do projeto):
    python src/testar_casos.py            # só as ferramentas
    python src/testar_casos.py --agente   # também a conversa com o agente (precisa de ANTHROPIC_API_KEY)

Gera:
  results/tables/casos_exemplo.csv   resumo dos 3 casos
  results/casos_exemplo.md           relatório legível (explicações e recomendações)
"""
import argparse
import os

import pandas as pd

from config import TABELAS
from descricoes import FEATURES
from explicar_shap import carregar_modelo, dados_divididos
from ferramentas import avaliar_referenciacao, explicar, prever_risco

ESPERADO = {
    "claramente DRC": lambda r: r["decisao_modelo"] == "DRC" and r["categoria"] == "elevado",
    "caso limite": lambda r: r["deferir_ao_medico"],
    "saudável": lambda r: r["decisao_modelo"] == "sem DRC" and r["categoria"] == "baixo",
}
DESCRICAO_ESPERADO = {
    "claramente DRC": "decisão DRC e categoria elevado",
    "caso limite": "deferido ao médico (zona de incerteza)",
    "saudável": "decisão sem DRC e categoria baixo",
}


def escolher_casos(X_te, y_te, modelo, limiar):
    prob = pd.Series(modelo.predict_proba(X_te)[:, 1], index=X_te.index)
    return {
        "claramente DRC": prob[y_te == 1].idxmax(),
        "caso limite": (prob - limiar).abs().idxmin(),
        "saudável": prob[y_te == 0].idxmin(),
    }


def dados_do_doente(linha):
    """Só os valores que existem (o médico não inventa os que faltam)."""
    return {v: float(linha[v]) for v in FEATURES if pd.notna(linha[v])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agente", action="store_true",
                        help="também pede ao agente Claude para avaliar cada caso")
    args = parser.parse_args()

    final = carregar_modelo()
    _, X_te, _, y_te = dados_divididos()
    casos = escolher_casos(X_te, y_te, final["modelo"], final["limiar"])
    print(f"Modelo: {final['nome']} | limiar {final['limiar']:.3f} | "
          f"probabilidade: {final.get('probabilidade', 'original')}\n")

    agente = None
    if args.agente:
        from agente import Agente, chave_disponivel
        if not chave_disponivel():
            raise SystemExit("Defina OPENROUTER_API_KEY (ou ANTHROPIC_API_KEY) para testar o agente (ver README).")

    linhas, relatorio, falhas = [], [], 0
    relatorio.append(f"# Casos de exemplo\n\nModelo: {final['nome']} · limiar "
                     f"{final['limiar']:.3f} · probabilidade {final.get('probabilidade', 'original')}\n")
    for tipo, idx in casos.items():
        dados = dados_do_doente(X_te.loc[idx])
        risco = prever_risco(dados)
        expl = explicar(dados)
        ref = avaliar_referenciacao(dados)
        ok = ESPERADO[tipo](risco)
        falhas += not ok

        print(f"=== {tipo.upper()} (doente {idx}, real: {'DRC' if y_te[idx] else 'sem DRC'}) ===")
        print(f"Probabilidade {risco['probabilidade_drc']:.1%} | decisão {risco['decisao_modelo']} | "
              f"categoria {risco['categoria']} | deferir ao médico: {risco['deferir_ao_medico']}")
        print(f"Esperado: {DESCRICAO_ESPERADO[tipo]} -> {'OK' if ok else 'FALHOU'}")
        print(f"Explicação: {expl['texto']}")
        print(f"Referenciação: {ref['recomendacao']}\n")

        linhas.append({
            "caso": tipo, "doente": idx, "real": "DRC" if y_te[idx] else "sem DRC",
            "probabilidade": risco["probabilidade_drc"], "decisao_modelo": risco["decisao_modelo"],
            "categoria": risco["categoria"], "deferir_ao_medico": risco["deferir_ao_medico"],
            "essenciais_em_falta": ", ".join(e["variavel"] for e in risco["essenciais_em_falta"]),
            "top5_shap": ", ".join(f["variavel"] for f in expl["fatores_principais"]),
            "referenciacao": ref["recomendacao"], "comportamento_esperado": DESCRICAO_ESPERADO[tipo],
            "ok": ok,
        })
        relatorio += [
            f"## {tipo} (doente {idx}, diagnóstico real: {'DRC' if y_te[idx] else 'sem DRC'})\n",
            f"- Probabilidade: {risco['probabilidade_drc']:.1%} · decisão: {risco['decisao_modelo']} · "
            f"categoria: {risco['categoria']} · deferir ao médico: {risco['deferir_ao_medico']}",
            f"- Esperado: {DESCRICAO_ESPERADO[tipo]} → **{'OK' if ok else 'FALHOU'}**",
            f"- Explicação: {expl['texto']}",
            f"- Referenciação: {ref['recomendacao']}",
            *[f"  - {c}" for c in ref["criterios_cumpridos"]],
            "",
        ]

        if args.agente:
            agente = Agente()   # conversa nova por caso
            pedido = (f"Avalie este doente (identificador exemplo-{idx}). "
                      f"Valores: {dados}")
            resposta, chamadas = agente.perguntar(pedido)
            usadas = [c["ferramenta"] for c in chamadas]
            print(f"Agente usou: {usadas}\nAgente respondeu:\n{resposta}\n")
            relatorio += [f"**Resposta do agente** (ferramentas: {', '.join(usadas)}):\n",
                          resposta, ""]

    relatorio.append("\n> Este sistema é apoio à decisão e não substitui a avaliação médica.\n")
    pd.DataFrame(linhas).to_csv(TABELAS / "casos_exemplo.csv", index=False, encoding="utf-8-sig")
    (TABELAS.parent / "casos_exemplo.md").write_text("\n".join(relatorio), encoding="utf-8")
    print(f"{3 - falhas} de 3 casos com o comportamento esperado.")
    print("Guardado: results/tables/casos_exemplo.csv e results/casos_exemplo.md")


if __name__ == "__main__":
    main()
