import streamlit as st
import joblib
import numpy as np

# 📌 Caminho do modelo salvo no Gihub
model_path = "modelo_credito_rf.pkl"

# 📌 Carregar o modelo treinado
modelo = joblib.load(model_path)

# 📌 Título da Aplicação
st.title("🚀 Sistema de Previsão de Crédito")

# 📌 Subtítulo e instruções
st.markdown("### Insira os dados do cliente para previsão:")
st.write("Preencha os campos abaixo e clique no botão para prever o risco de crédito.")

# 📌 Criar campos de entrada para os dados do cliente
idade = st.number_input("Idade", min_value=18, max_value=100, value=30, step=1)
tempo_emprego = st.number_input("Tempo de Emprego (anos)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
qtd_filhos = st.number_input("Quantidade de Filhos", min_value=0, max_value=10, value=1, step=1)
renda = st.number_input("Renda Mensal", min_value=500, max_value=100000, value=5000, step=100)

# 📌 Criar botão para fazer a previsão
if st.button("📊 Fazer Previsão"):
    entrada = np.array([[idade, tempo_emprego, qtd_filhos, renda]])

# 📌 Verificar número de features antes da previsão
    if entrada.shape[1] != modelo.n_features_in_:
        st.error(f"Erro: O modelo espera {modelo.n_features_in_} features, mas recebeu {entrada.shape[1]}.")
    else:
        score = modelo.predict(entrada)[0]
        st.metric(label="Score do Cliente", value=round(score, 3))
   

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
