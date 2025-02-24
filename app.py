import streamlit as st
import joblib
import numpy as np

# 📌 Carregar o modelo
modelo = joblib.load("modelo_credito_rf.pkl")

# 📌 Título da Aplicação
st.title("🚀 Sistema de Previsão de Crédito")

st.markdown("### Insira os dados do cliente para previsão:")
st.write("Preencha os campos abaixo e clique no botão para prever o risco de crédito.")

# 📌 Criar campos para TODAS as features esperadas pelo modelo
posse_de_veiculo = st.checkbox("Possui Veículo?")
posse_de_imovel = st.checkbox("Possui Imóvel?")
qtd_filhos = st.number_input("Quantidade de Filhos", min_value=0, max_value=10, value=1, step=1)
idade = st.number_input("Idade", min_value=18, max_value=100, value=30, step=1)
tempo_emprego = st.number_input("Tempo de Emprego (anos)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
qt_pessoas_residencia = st.number_input("Quantidade de Pessoas na Residência", min_value=1, max_value=10, value=3, step=1)
renda = st.number_input("Renda Mensal", min_value=500, max_value=100000, value=5000, step=100)

# 📌 Variáveis categóricas convertidas para dummies
tipo_renda = st.selectbox("Tipo de Renda", ["Bolsista", "Empresário", "Pensionista", "Servidor público"])
educacao = st.selectbox("Nível Educacional", ["Pós graduação", "Secundário", "Superior completo", "Superior incompleto"])
estado_civil = st.selectbox("Estado Civil", ["Separado", "Solteiro", "União", "Viúvo"])
tipo_residencia = st.selectbox("Tipo de Residência", ["Casa", "Com os pais", "Comunitário", "Estúdio", "Governamental"])
sexo_M = st.radio("Sexo", ["Feminino", "Masculino"]) == "Masculino"

# 📌 Criar faixas etárias e categorias de tempo de emprego
faixa_etaria_Adulto = 18 <= idade < 50
faixa_etaria_Idoso = idade >= 50
tempo_emprego_cat_5_10_anos = 5 <= tempo_emprego < 10
tempo_emprego_cat_mais_10_anos = tempo_emprego >= 10

# 📌 Mapear as seleções para variáveis binárias
tipo_renda_map = {
    "Bolsista": [1, 0, 0, 0],
    "Empresário": [0, 1, 0, 0],
    "Pensionista": [0, 0, 1, 0],
    "Servidor público": [0, 0, 0, 1],
}
educacao_map = {
    "Pós graduação": [1, 0, 0, 0],
    "Secundário": [0, 1, 0, 0],
    "Superior completo": [0, 0, 1, 0],
    "Superior incompleto": [0, 0, 0, 1],
}
estado_civil_map = {
    "Separado": [1, 0, 0, 0],
    "Solteiro": [0, 1, 0, 0],
    "União": [0, 0, 1, 0],
    "Viúvo": [0, 0, 0, 1],
}
tipo_residencia_map = {
    "Casa": [1, 0, 0, 0, 0],
    "Com os pais": [0, 1, 0, 0, 0],
    "Comunitário": [0, 0, 1, 0, 0],
    "Estúdio": [0, 0, 0, 1, 0],
    "Governamental": [0, 0, 0, 0, 1],
}

# 📌 Criar array de entrada com TODAS as features esperadas pelo modelo
entrada = np.array([[
    posse_de_veiculo, posse_de_imovel, qtd_filhos, idade, tempo_emprego,
    qt_pessoas_residencia, renda,
    *tipo_renda_map[tipo_renda], *educacao_map[educacao], *estado_civil_map[estado_civil],
    *tipo_residencia_map[tipo_residencia], sexo_M, faixa_etaria_Adulto, faixa_etaria_Idoso,
    tempo_emprego_cat_5_10_anos, tempo_emprego_cat_mais_10_anos
]])

# 📌 Verificar número de features antes da previsão
if entrada.shape[1] != modelo.n_features_in_:
    st.error(f"Erro: O modelo espera {modelo.n_features_in_} features, mas recebeu {entrada.shape[1]}.")
else:
    score = modelo.predict(entrada)[0]
    
    # 📌 Aplicar regra de decisão
    if score >= 0.6:
        decisao = "✅ **Aprovado Automaticamente**"
    elif score >= 0.4:
        decisao = "🔍 **Encaminhado para Análise Manual**"
    else:
        decisao = "❌ **Negado Automaticamente**"

    # 📌 Exibir Resultado
    st.markdown(f"## 📢 Decisão: {decisao}")
    st.metric(label="Score do Cliente", value=round(score, 3))
