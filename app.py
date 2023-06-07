import pandas as pd
from pycaret.regression import *
import streamlit as st

# Carregar os dados do arquivo CSV
url = 'https://dados.agricultura.gov.br/dataset/58bdc09c-9778-42b9-8fce-7d5c2c4fa211/resource/b89ffa6b-cf9a-4fb6-9f94-c9fcbfc0a6d7/download/imoveis_cadastrados_por_municipio.csv'
df = pd.read_csv(url, sep=';')

# Remover a coluna 'municipio'
df.drop(columns=['municipio'], inplace=True)

# Corrigir tipos de dados
df['codigo_ibge'] = df['codigo_ibge'].astype(int)
df['numero_de_cadastros'] = df['numero_de_cadastros'].astype(int)
df['area_cadastrada'] = df['area_cadastrada'].astype(float)

# Inicializar o ambiente do PyCaret
regression_setup = setup(data=df, target='numero_de_cadastros')

# Selecionar um modelo de regressão
regression_model = create_model('lr')  # lr = Linear Regression

# Treinar o modelo
trained_model = finalize_model(regression_model)

# Função para fazer a previsão
def fazer_previsao(input1, input2):
    input_df = pd.DataFrame({'uf': ['SP'], 'codigo_ibge': [input1], 'area_cadastrada': [input2]})
    prediction = predict_model(trained_model, data=input_df)
    label_column = prediction.columns[-1]  # Pegar o nome da última coluna (pode variar de acordo com a versão do PyCaret)
    return prediction[label_column].iloc[0]

# Configurar o estilo da página
st.set_page_config(page_title="Previsão de Cadastro em Imóveis", layout="centered")

# Título do webapp
st.title("Previsão de Cadastro em Imóveis")

# Descrição do webapp
st.write("Insira os dados necessários para fazer a previsão.")

# Layout em duas colunas
col1, col2 = st.columns(2)

# Campo de entrada para Código IBGE
with col1:
    input1 = st.number_input("Código IBGE")

# Campo de entrada para Área Cadastrada
with col2:
    input2 = st.number_input("Área Cadastrada")

# Botão de predição
if st.button("Fazer Previsão"):
    prediction = fazer_previsao(input1, input2)
    prediction_rounded = round(prediction, 2)  # Arredonda para 2 casas decimais
    st.success("Previsão: {}".format(prediction_rounded))

