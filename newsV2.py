import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas_ta as ta
import matplotlib.pyplot as plt
import os

# Configuração do Streamlit
st.title('Previsão de Preços de Ações')
st.write('Este aplicativo prevê os preços futuros das ações usando um modelo de Random Forest.')

# Campo de entrada para o ticker
ticker = st.text_input('Digite o Ticker da Ação', value='BBAS3.SA')

# Seletores de data para start_date e end_date
start_date = st.date_input('Data de Início', value=pd.to_datetime('2000-01-01'))
end_date = st.date_input('Data de Fim', value=pd.to_datetime('2024-07-10'))

# Campo de entrada para o número de dias de previsão
num_days_forecast = st.number_input('Número de Dias para Prever', min_value=1, max_value=30, value=5)

# Seletores de data para start_date_target e end_date_target
start_date_target = st.date_input('Data de Início para Análise de Proximidade', value=pd.to_datetime('2024-07-07'))
end_date_target = st.date_input('Data de Fim para Análise de Proximidade', value=pd.to_datetime('2024-07-12'))

# Botão para iniciar a previsão
if st.button('Pesquisar'):
    # Coleta de dados
    data = yf.download(ticker, start=start_date, end=end_date)

    # Verificar se o índice está no formato DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Feature Engineering
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    bbands = ta.bbands(data['Close'], length=20)
    data['BB_upper'] = bbands['BBU_20_2.0']
    data['BB_middle'] = bbands['BBM_20_2.0']
    data['BB_lower'] = bbands['BBL_20_2.0']

    # Criar coluna de preço futuro (1 dia à frente)
    data['Future_Close'] = data['Close'].shift(-1)
    data = data.dropna()

    # Preparação dos dados
    features = ['SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']
    X = data[features]
    y = data['Future_Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelagem
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Função para recalcular as features
    def recalculate_features(temp_series):
        temp_df = pd.Series(temp_series)
        sma_50 = ta.sma(temp_df, length=50).iloc[-1] if len(temp_df) >= 50 else np.nan
        rsi = ta.rsi(temp_df, length=14).iloc[-1] if len(temp_df) >= 14 else np.nan
        macd = ta.macd(temp_df).iloc[-1]['MACD_12_26_9'] if len(temp_df) >= 26 else np.nan
        bbands = ta.bbands(temp_df, length=20).iloc[-1] if len(temp_df) >= 20 else {'BBU_20_2.0': np.nan, 'BBM_20_2.0': np.nan, 'BBL_20_2.0': np.nan}

        return [sma_50, rsi, macd, bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']]

    # Previsão Recursiva para os Próximos Dias
    last_data = data.iloc[-1][features].values.reshape(1, -1)
    future_predictions = []

    # Série temporária para armazenar previsões
    temp_series = data['Close'].tolist()

    for _ in range(num_days_forecast):
        next_pred = model.predict(pd.DataFrame(last_data, columns=features))[0]
        future_predictions.append(next_pred)

        # Atualizar a série temporária com a nova previsão
        temp_series.append(next_pred)

        # Recalcular as features com base na série temporária atualizada
        last_data = pd.DataFrame([recalculate_features(temp_series)], columns=features)

    # Visualização das Previsões
    st.subheader('Previsão de Preços')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[-50:], data['Close'][-50:], label='Histórico')
    future_dates = pd.date_range(start=data.index[-1], periods=num_days_forecast + 1, freq='B')[1:]  # 'B' para dias úteis
    ax.plot(future_dates, future_predictions, label='Previsão', linestyle='--')

    # Adicionar anotações de preços e setas indicando subida ou queda
    for i, txt in enumerate(future_predictions):
        ax.annotate(f'{txt:.2f}', (future_dates[i], future_predictions[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        if i > 0:
            if future_predictions[i] > future_predictions[i-1]:
                ax.annotate('↑', (future_dates[i], future_predictions[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=15, color='green')
            else:
                ax.annotate('↓', (future_dates[i], future_predictions[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=15, color='red')

    # Criar tabela com as previsões
    table_data = {
        'Data': future_dates,
        'Preço Previsto': [f'{pred:.2f}' for pred in future_predictions]
    }
    table_df = pd.DataFrame(table_data)

    # Adicionar tabela ao gráfico
    ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.3])

    ax.set_xlabel('Data')
    ax.set_ylabel('Preço de Fechamento')
    ax.set_title(f'Previsão de Preço para os Próximos {num_days_forecast} Dias')
    ax.legend()
    plt.subplots_adjust(bottom=0.3)  # Ajustar para dar espaço à tabela
    plt.savefig('previsao_dias.png')  # Salvar o gráfico como imagem
    st.pyplot(fig)

    # Verificar se o arquivo foi salvo corretamente
    if os.path.exists('previsao_dias.png'):
        st.success("Gráfico salvo com sucesso.")
    else:
        st.error("Erro ao salvar o gráfico.")

    # Parte 2: Análise de Proximidade ao Preço Alvo
    target_price = future_predictions[-1]  # Definir o preço alvo como o último preço previsto

    # Baixar os dados
    data_target = yf.download(ticker, start=start_date_target, end=end_date_target)

    # Verificar se há dados disponíveis
    if data_target.empty:
        st.warning("Não foram encontrados dados para o período especificado.")
    else:
        # Adicionar uma coluna com o preço alvo
        data_target['Target'] = target_price

        # Calcular a proximidade ao preço alvo
        data_target['Proximidade (%)'] = (data_target[['Close', 'Target']].min(axis=1) /
                                          data_target[['Close', 'Target']].max(axis=1)) * 100

        # Adicionar coluna de diferença entre Close e Target
        data_target['Diferença (Close - Previsão)'] = data_target['Close'] - data_target['Target']

        # Formatar as colunas Close e Diferença (Close - Previsão) com R$
        data_target['Close'] = data_target['Close'].apply(lambda x: f'R$ {x:.2f}')
        data_target['Diferença (Close - Previsão)'] = data_target['Diferença (Close - Previsão)'].apply(lambda x: f'R$ {x:.2f}')

        # Exibir os preços de fechamento, a proximidade ao preço alvo e a diferença
        st.subheader('Análise de Proximidade ao Preço Alvo')
        st.write(data_target[['Close', 'Diferença (Close - Previsão)', 'Proximidade (%)']])

        # Plotar o gráfico
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Gráfico do preço de fechamento
        ax[0].plot(data_target.index, data_target['Close'].str.replace('R$', '').astype(float), marker='o', linestyle='-', color='b', label='Preço de Fechamento')
        ax[0].axhline(y=target_price, color='r', linestyle='--', label=f'Preço Alvo ({target_price:.2f})')
        ax[0].set_title(f'Preço de Fechamento de {ticker} de {start_date_target} a {end_date_target}')
        ax[0].set_xlabel('Data')
        ax[0].set_ylabel('Preço de Fechamento (R$)')
        ax[0].legend()
        ax[0].grid(True)

        # Adicionar a tabela de preços de fechamento, proximidade e diferença ao gráfico
        table_data_target = data_target[['Close', 'Diferença (Close - Previsão)', 'Proximidade (%)']].reset_index()
        table_data_target['Date'] = table_data_target['Date'].dt.strftime('%Y-%m-%d')
        table_data_target = table_data_target[['Date', 'Close', 'Diferença (Close - Previsão)', 'Proximidade (%)']]

        # Adicionar a linha do preço alvo
        target_row = pd.DataFrame([['Preço Alvo', f'R$ {target_price:.2f}', '', '']], columns=table_data_target.columns)
        table_data_target = pd.concat([target_row, table_data_target], ignore_index=True)

        ax[1].axis('tight')
        ax[1].axis('off')
        table = ax[1].table(cellText=table_data_target.values, colLabels=table_data_target.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.tight_layout()

        # Exibir o gráfico
        st.pyplot(fig)