from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("modelo.pkl")

st.set_page_config(page_title="Previsão Iris", page_icon="🌸", layout="centered")


@st.cache_resource
def carregar_modelo():
    if not MODEL_PATH.exists():
        st.error("Arquivo modelo.pkl não encontrado. Execute primeiro: python train.py")
        st.stop()
    return joblib.load(MODEL_PATH)


artefato = carregar_modelo()
model = artefato["model"]
features = artefato["features"]
accuracy = artefato["accuracy"]


st.title("🌸 Classificador de Flores Iris")
st.write("Informe as medidas da flor para o modelo prever a espécie em tempo real.")
st.info(f"Acurácia no teste: {accuracy:.2%}")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Comprimento da sépala (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Largura da sépala (cm)", 2.0, 4.5, 3.5, 0.1)

with col2:
    petal_length = st.slider("Comprimento da pétala (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Largura da pétala (cm)", 0.1, 2.5, 0.2, 0.1)

entrada = pd.DataFrame([
    {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
])[features]

previsao = model.predict(entrada)[0]
probabilidades = model.predict_proba(entrada)[0]
classes = model.classes_

st.subheader("Resultado")
st.success(f"Espécie prevista: **{previsao}**")

st.subheader("Probabilidades")
prob_df = pd.DataFrame({"Espécie": classes, "Probabilidade": probabilidades})
st.bar_chart(prob_df.set_index("Espécie"))

with st.expander("Ver dados enviados ao modelo"):
    st.dataframe(entrada)
