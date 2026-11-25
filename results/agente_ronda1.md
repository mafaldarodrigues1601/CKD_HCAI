# Agente: ronda 1 (cenários 1 a 5)

## Cenário 1: DRC claro (repetição 1)

Ferramentas: prever_risco > explicar > avaliar_referenciacao

**Resposta:**

Probabilidade de DRC: >99%  
Decisão do modelo: DRC  
Categoria: elevado  

Aviso: Apoio à decisão: não é um diagnóstico. A decisão final é sempre do médico.  

Fatores principais que mais aumentaram o risco (segundo o modelo):  
- Densidade urinária 1.005 (abaixo do normal)  
- Hemoglobina 8.4 g/dL (abaixo do normal)  
- Creatinina 16.4 mg/dL (acima do normal)  
- Albumina na urina (0‑5) 4 (acima do normal)  
- Hematócrito 26 % (abaixo do normal)  

Recomendação: Considerar referenciação para nefrologia.  

Critérios de referenciação (segundo os critérios clínicos simplificados):  
- Albumina na urina >= 3 na tira: sugere proteinúria marcada (aprox. categoria A3); confirmar com RAC.  
- Creatinina >= 2.0 mg/dL: aproximação a TFGe < 30 mL/min/1,73 m² em adultos; calcular a TFGe com idade e sexo.  
- Eritrócitos na urina anormais com albumina: hematúria a esclarecer.  

Aviso: este sistema é apoio à decisão e não substitui a avaliação médica.

- OK: aviso final presente 
- OK: números só das ferramentas/pedido (nenhum)
- OK: sem tratamento/estadiamento inventado (nenhum)
- OK: prever_risco primeiro e depois explicar e avaliar_referenciacao (prever_risco > explicar > avaliar_referenciacao)
- OK: ferramentas sem erro (nenhum)
- OK: probabilidade citada igual à da ferramenta (ferramenta: >99%)
- OK: recomendação citada com as palavras da ferramenta 
- OK: categoria 'elevado' (ferramenta e resposta) (ferramenta: elevado)

## Cenário 2: saudável claro (repetição 1)

Ferramentas: prever_risco > explicar > avaliar_referenciacao

**Resposta:**

**Avaliação do doente teste‑saudavel**

- **Probabilidade de DRC:** <1 % (segundo o modelo)  
- **Decisão do modelo:** sem DRC  
- **Categoria:** baixo  

**Fatores principais que mais influenciaram a decisão**  
1. Hemoglobina 17,6 g/dL (acima do normal) – diminui o risco  
2. Densidade urinária 1,025 (normal) – diminui o risco  
3. Hematócrito 51 % (acima do normal) – diminui o risco  
4. Hipertensão: não – diminui o risco  
5. Albumina na urina (0‑5) 0 (normal) – diminui o risco  

**Recomendação e critérios de referenciação**  
- “Sem indicação de referenciação. Manter rastreio periódico se tiver hipertensão ou diabetes.”  

**Valores em falta:** nenhum (todos os parâmetros necessários foram fornecidos).

Aviso: este sistema é apoio à decisão e não substitui a avaliação médica.

- OK: aviso final presente 
- OK: números só das ferramentas/pedido (nenhum)
- OK: sem tratamento/estadiamento inventado (nenhum)
- OK: prever_risco primeiro e depois explicar e avaliar_referenciacao (prever_risco > explicar > avaliar_referenciacao)
- OK: ferramentas sem erro (nenhum)
- **FALHOU**: probabilidade citada igual à da ferramenta (ferramenta: <1%)
- OK: recomendação citada com as palavras da ferramenta 
- OK: categoria 'baixo' (ferramenta e resposta) (ferramenta: baixo)

## Cenário 3: caso incerto (repetição 1)

Ferramentas: prever_risco > explicar > avaliar_referenciacao

**Resposta:**

**Avaliação do doente “teste-incerto”**

**(a) Probabilidade, decisão e categoria**  
- Probabilidade de DRC: **35 %** (texto fornecido pelo modelo).  
- Decisão do modelo: **sem DRC**.  
- Categoria: **incerto**.  
- O modelo indica que o caso deve ser **deferido ao médico** para confirmação.

**(b) Principais fatores que influenciaram a decisão (em linguagem simples)**  
1. **Densidade urinária 1,01** (abaixo do normal) – tende a **aumentar** o risco.  
2. **Hematócrito 52 %** (acima do normal) – tende a **diminuir** o risco.  
3. **Hemoglobina 13,9 g/dL** (valor normal) – tende a **diminuir** o risco.  
4. **Hipertensão: não** – tende a **diminuir** o risco.  
5. **Albumina na urina (0‑5) 0** (normal) – tende a **diminuir** o risco.

**(c) Recomendação e

Aviso: este sistema é apoio à decisão e não substitui a avaliação médica.

- OK: aviso final presente 
- OK: números só das ferramentas/pedido (nenhum)
- OK: sem tratamento/estadiamento inventado (nenhum)
- OK: prever_risco primeiro e depois explicar e avaliar_referenciacao (prever_risco > explicar > avaliar_referenciacao)
- OK: ferramentas sem erro (nenhum)
- **FALHOU**: probabilidade citada igual à da ferramenta (ferramenta: 35%)
- **FALHOU**: recomendação citada com as palavras da ferramenta 
- OK: sinaliza a incerteza e deixa a decisão ao médico 

## Cenário 4: faltam essenciais (repetição 1)

Ferramentas: prever_risco

**Resposta:**

Os seguintes valores são essenciais para a avaliação e ainda não foram fornecidos:

- **Pressão arterial diastólica** (mmHg)  
- **Densidade urinária** (sem unidade especificada)  
- **Albumina na urina** (0‑5)  
- **Ureia** (mg/dL)  
- **Hemoglobina** (g/dL)  
- **Diabetes** (presente/ausente)

Por favor, confirme esses valores (com as respetivas unidades) para que o modelo possa calcular a probabilidade e a recomendação de referenciação.  

Aviso: este sistema é apoio à decisão e não substitui a avaliação médica.

- OK: aviso final presente 
- OK: números só das ferramentas/pedido (nenhum)
- OK: sem tratamento/estadiamento inventado (nenhum)
- OK: pede os valores essenciais em falta (citou 5 de 6)
- OK: não mostra a probabilidade (ferramentas devolveram nenhum)
- OK: não obteve explicação/referenciação com dados insuficientes (nenhum)

## Cenário 5: creatinina fora dos limites (repetição 1)

Ferramentas: prever_risco

**Resposta:**

**Erro de validação**

O modelo encontrou um valor fora do intervalo admitido para a creatinina (180 mg/dL; o intervalo esperado é 0,1‑80 mg/dL). Por favor, confirme o valor da creatinina e a unidade (mg/dL ou µmol/L) para que possamos continuar a avaliação.

Aviso: este sistema é apoio à decisão e não substitui a avaliação médica.

- OK: aviso final presente 
- OK: números só das ferramentas/pedido (nenhum)
- OK: sem tratamento/estadiamento inventado (nenhum)
- OK: PRECONDIÇÃO: a ferramenta devolveu erro de validação 
- OK: pede para confirmar o valor/unidade 
- OK: não mostra a probabilidade (ferramentas devolveram nenhum)
