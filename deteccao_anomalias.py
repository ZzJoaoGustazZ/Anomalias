"""
Detecção de Anomalias em Transações Financeiras
=================================================
Laboratório: Detecção de Anomalias em Transações (DIO)
Abordagem: Estatística — Z-score (desvio padrão)

O que este script faz:
1. Gera (ou carrega) uma base de transações financeiras.
2. Calcula o Z-score do valor de cada transação.
3. Classifica como ANOMALIA toda transação cujo Z-score absoluto
   ultrapasse um limiar (threshold) definido — por padrão, 3.
4. Também aplica uma análise complementar por conta (cada cliente
   tem seu próprio padrão de gasto), pois um valor "normal" para
   um cliente pode ser uma anomalia para outro.
5. Exporta um relatório em CSV com as transações sinalizadas e
   gera um gráfico com a distribuição dos valores e os pontos
   fora do padrão.

Como usar com seus próprios dados:
- Se você tiver um CSV real, basta chamar:
      carregar_dados("caminho/para/seu_arquivo.csv")
  O arquivo precisa ter, no mínimo, as colunas:
      id_transacao, id_conta, valor, data_hora
- Caso não informe um arquivo, o script gera dados sintéticos
  automaticamente para fins de teste/demonstração.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------
SEED = 42
Z_THRESHOLD = 3.0          # limiar global de anomalia (em desvios padrão)
Z_THRESHOLD_POR_CONTA = 2.5  # limiar mais sensível, por conta individual
OUTPUT_DIR = Path(__file__).parent
np.random.seed(SEED)


# ---------------------------------------------------------------
# 1) Geração / carregamento dos dados
# ---------------------------------------------------------------
def gerar_dados_sinteticos(n_transacoes: int = 2000, n_contas: int = 50) -> pd.DataFrame:
    """
    Gera uma base sintética de transações financeiras, simulando
    o comportamento normal de clientes e injetando algumas
    transações anômalas (valores muito altos ou muito fora do padrão).
    """
    contas = [f"CTA-{i:04d}" for i in range(n_contas)]

    # cada conta tem um "perfil" de gasto médio diferente
    perfil_conta = {c: np.random.uniform(50, 500) for c in contas}

    registros = []
    data_base = pd.Timestamp("2026-01-01")

    for i in range(n_transacoes):
        conta = np.random.choice(contas)
        media = perfil_conta[conta]
        # valores normais seguem uma distribuição próxima da média da conta
        valor = np.random.normal(loc=media, scale=media * 0.15)
        valor = max(valor, 1.0)

        data_hora = data_base + pd.Timedelta(
            days=np.random.randint(0, 180),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )

        registros.append({
            "id_transacao": f"TX-{i:06d}",
            "id_conta": conta,
            "valor": round(valor, 2),
            "data_hora": data_hora,
        })

    df = pd.DataFrame(registros)

    # injeta anomalias propositais (fraudes simuladas): ~2% das transações
    n_anomalias = int(n_transacoes * 0.02)
    indices_anomalos = np.random.choice(df.index, size=n_anomalias, replace=False)
    for idx in indices_anomalos:
        conta = df.loc[idx, "id_conta"]
        media = perfil_conta[conta]
        # valor muito acima do padrão da conta (ex.: 8x a 20x a média)
        df.loc[idx, "valor"] = round(media * np.random.uniform(8, 20), 2)

    return df.sort_values("data_hora").reset_index(drop=True)


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega uma base real de transações a partir de um CSV.
    Espera colunas: id_transacao, id_conta, valor, data_hora
    """
    df = pd.read_csv(caminho_csv, parse_dates=["data_hora"])
    colunas_esperadas = {"id_transacao", "id_conta", "valor", "data_hora"}
    faltando = colunas_esperadas - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas faltando no CSV: {faltando}")
    return df


# ---------------------------------------------------------------
# 2) Cálculo do Z-score e detecção de anomalias
# ---------------------------------------------------------------
def calcular_zscore(serie: pd.Series) -> pd.Series:
    """Calcula o Z-score de uma série numérica: (x - média) / desvio padrão."""
    media = serie.mean()
    desvio = serie.std(ddof=0)
    if desvio == 0:
        return pd.Series(np.zeros(len(serie)), index=serie.index)
    return (serie - media) / desvio


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica duas camadas de detecção:
      - Z-score global (todas as transações juntas)
      - Z-score por conta (compara cada transação apenas com o
        histórico daquela conta específica), que é mais realista
        pois cada cliente tem um padrão de consumo diferente.
    Uma transação é marcada como anômala se qualquer uma das duas
    regras disparar.
    """
    df = df.copy()

    # Z-score global
    df["zscore_global"] = calcular_zscore(df["valor"])

    # Z-score por conta
    df["zscore_conta"] = df.groupby("id_conta")["valor"].transform(calcular_zscore)

    df["anomalia_global"] = df["zscore_global"].abs() > Z_THRESHOLD
    df["anomalia_por_conta"] = df["zscore_conta"].abs() > Z_THRESHOLD_POR_CONTA

    df["anomalia"] = df["anomalia_global"] | df["anomalia_por_conta"]

    return df


# ---------------------------------------------------------------
# 3) Relatório e visualização
# ---------------------------------------------------------------
def gerar_relatorio(df: pd.DataFrame) -> pd.DataFrame:
    anomalias = df[df["anomalia"]].sort_values("valor", ascending=False)

    print("=" * 60)
    print("RELATÓRIO DE DETECÇÃO DE ANOMALIAS")
    print("=" * 60)
    print(f"Total de transações analisadas : {len(df)}")
    print(f"Total de anomalias detectadas   : {len(anomalias)} "
          f"({len(anomalias) / len(df):.2%})")
    print(f"Valor médio das transações      : R$ {df['valor'].mean():,.2f}")
    print(f"Valor médio das anomalias       : R$ {anomalias['valor'].mean():,.2f}" if len(anomalias) else "")
    print("-" * 60)
    if len(anomalias):
        print("Top 10 transações mais suspeitas:")
        colunas_exibir = ["id_transacao", "id_conta", "valor", "data_hora",
                           "zscore_global", "zscore_conta"]
        print(anomalias[colunas_exibir].head(10).to_string(index=False))
    print("=" * 60)

    caminho_saida = OUTPUT_DIR / "anomalias_detectadas.csv"
    anomalias.to_csv(caminho_saida, index=False)
    print(f"\nRelatório completo salvo em: {caminho_saida}")

    return anomalias


def plotar_grafico(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: distribuição dos valores com anomalias destacadas
    normais = df[~df["anomalia"]]
    anomalas = df[df["anomalia"]]

    axes[0].scatter(normais["data_hora"], normais["valor"],
                     alpha=0.4, s=15, label="Normal", color="#4C72B0")
    axes[0].scatter(anomalas["data_hora"], anomalas["valor"],
                     alpha=0.9, s=40, label="Anomalia", color="#C44E52",
                     marker="x")
    axes[0].set_title("Transações ao longo do tempo")
    axes[0].set_xlabel("Data/Hora")
    axes[0].set_ylabel("Valor (R$)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # Gráfico 2: histograma dos Z-scores globais
    axes[1].hist(df["zscore_global"], bins=40, color="#4C72B0", alpha=0.7)
    axes[1].axvline(Z_THRESHOLD, color="#C44E52", linestyle="--",
                     label=f"Limiar (+{Z_THRESHOLD})")
    axes[1].axvline(-Z_THRESHOLD, color="#C44E52", linestyle="--",
                     label=f"Limiar (-{Z_THRESHOLD})")
    axes[1].set_title("Distribuição dos Z-scores")
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    plt.tight_layout()
    caminho_grafico = OUTPUT_DIR / "grafico_anomalias.png"
    plt.savefig(caminho_grafico, dpi=150)
    print(f"Gráfico salvo em: {caminho_grafico}")


# ---------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------
def main(caminho_csv: str | None = None):
    if caminho_csv:
        df = carregar_dados(caminho_csv)
    else:
        print("Nenhum arquivo informado — gerando base sintética de teste...\n")
        df = gerar_dados_sinteticos()

    df_resultado = detectar_anomalias(df)
    gerar_relatorio(df_resultado)
    plotar_grafico(df_resultado)

    # salva também a base completa já com as colunas de análise
    caminho_completo = OUTPUT_DIR / "transacoes_analisadas.csv"
    df_resultado.to_csv(caminho_completo, index=False)
    print(f"Base completa (com scores) salva em: {caminho_completo}")


if __name__ == "__main__":
    import sys
    arquivo = sys.argv[1] if len(sys.argv) > 1 else None
    main(arquivo)
