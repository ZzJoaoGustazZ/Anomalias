# Detecção de Anomalias em Transações Financeiras

## 📌 Propósito

Este projeto tem como objetivo **identificar transações financeiras que fogem do padrão**, ajudando a encontrar possíveis operações suspeitas ou incomuns.

A aplicação analisa os valores das transações e identifica quando uma operação apresenta um comportamento muito diferente do que normalmente acontece.

## 🧠 Algoritmo utilizado

Utilizei o **Z-score** para identificar as anomalias.

De forma simples, o algoritmo verifica **o quanto uma transação está distante do valor considerado normal**. Quando uma transação apresenta um valor muito diferente das demais, ela é sinalizada como uma possível anomalia.

A implementação e a parte técnica do algoritmo estão disponíveis diretamente no código do projeto.

## 🛠️ Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib

## 📊 Resultado

O sistema identifica as transações consideradas anômalas e gera arquivos e gráficos para facilitar a visualização dos resultados.

> **Observação:** uma anomalia não significa necessariamente uma fraude. Ela apenas indica uma transação que apresenta um comportamento diferente do padrão.
