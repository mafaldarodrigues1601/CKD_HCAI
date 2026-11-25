# Casos de exemplo

Modelo: Regressão Logística (baseline) · limiar 0.642 · probabilidade original

## claramente DRC (doente 154, diagnóstico real: DRC)

- Probabilidade: 100.0% · decisão: DRC · categoria: elevado · deferir ao médico: False
- Esperado: decisão DRC e categoria elevado → **OK**
- Explicação: O modelo estima uma probabilidade de DRC de >99%. O que mais aumentou o risco: densidade urinária 1.005 (abaixo do normal); hemoglobina 8.4 g/dL (abaixo do normal); creatinina 16.4 mg/dL (acima do normal); albumina na urina (0-5) 4 (acima do normal); hematócrito 26 % (abaixo do normal). (Fatores por ordem de peso na decisão do modelo.) Em falta e substituídos por valores típicos do treino: leucócitos.
- Referenciação: Considerar referenciação para nefrologia.
  - Albumina na urina >= 3 na tira: sugere proteinúria marcada (aprox. categoria A3); confirmar com RAC.
  - Creatinina >= 2.0 mg/dL: aproximação a TFGe < 30 mL/min/1,73 m² em adultos; calcular a TFGe com idade e sexo.
  - Eritrócitos na urina anormais com albumina: hematúria a esclarecer.

## caso limite (doente 224, diagnóstico real: DRC)

- Probabilidade: 54.2% · decisão: sem DRC · categoria: incerto · deferir ao médico: True
- Esperado: deferido ao médico (zona de incerteza) → **OK**
- Explicação: O modelo estima uma probabilidade de DRC de 54%. O que mais aumentou o risco: edema dos membros inferiores: sim. O que mais o diminuiu: hipertensão: não; albumina na urina (0-5) 0 (normal); densidade urinária 1.02 (normal); diabetes: não. (Fatores por ordem de peso na decisão do modelo.) Em falta e substituídos por valores típicos do treino: hemoglobina, hematócrito, leucócitos, eritrócitos, eritrócitos na urina anormais.
- Referenciação: Considerar referenciação para nefrologia.
  - Creatinina >= 2.0 mg/dL: aproximação a TFGe < 30 mL/min/1,73 m² em adultos; calcular a TFGe com idade e sexo.

## saudável (doente 371, diagnóstico real: sem DRC)

- Probabilidade: 0.0% · decisão: sem DRC · categoria: baixo · deferir ao médico: False
- Esperado: decisão sem DRC e categoria baixo → **OK**
- Explicação: O modelo estima uma probabilidade de DRC de <1%. O que mais o diminuiu: hemoglobina 17.6 g/dL (acima do normal); densidade urinária 1.025 (normal); hematócrito 51 % (acima do normal); hipertensão: não; albumina na urina (0-5) 0 (normal). (Fatores por ordem de peso na decisão do modelo.)
- Referenciação: Sem indicação de referenciação. Manter rastreio periódico se tiver hipertensão ou diabetes.


> Este sistema é apoio à decisão e não substitui a avaliação médica.
