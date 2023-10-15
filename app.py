import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps
import numpy as np
import cv2
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.cloud import storage
from explain import main_funct

from helpers import *

######################################################################################

# Funcionalidad almacenamiento de los datos

creds = Credentials.from_service_account_file('acquired-winter-316123-902a50e3c0d5.json')
bq_client = bigquery.Client(credentials=creds)
storage_client = storage.Client(credentials=creds)

TABLA_PACIENTE = 'acquired-winter-316123.capstone.paciente'


def save_image(image):
    file_name = image.name
    bucket = storage_client.bucket('melanoma_capstone_bucket')
    blob = bucket.blob(file_name)
    try:
        blob.upload_from_file(file_obj=image, rewind=True)
    except Exception as e:
        st.text(e)
    else:
        st.text(blob.public_url)


def insert_patient():
    pat_dict = {'paciente_id': 1, 
                'edad': edad,
                'sexo': sexo,
                'canc_ant': True if canc_ant == "Si" else False,
                'canc_pred': True if prediction >= 0.5 else False,
                'cancer_diag': True if canc_diag == "Maligno" else False}
    try:
        bq_client.insert_rows_json(TABLA_PACIENTE, [pat_dict])
    except Exception as e:
        st.error(e)
    else:
        st.success('Datos Guardados')

#####################################################################################



prediction = None

st.set_page_config(page_title="Melanoma AI", page_icon="🔬")

st.markdown(
    f"""
    <style>
        img {{
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Función de carga del model
@st.cache_resource
def load_model_from_file():
    return load_model('model_daugmentation.h5')

model = load_model_from_file()


# Procesamiento Imágen subida por usuario
def preproc_img(image):
    img = load_img(image, target_size=(300, 300))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

st.title("AI para la detección del cáncer de piel")

# Widget para subir archivo a procesar
img = st.file_uploader("Selecciona la imágen a evaluar", type=None, label_visibility="visible")


# Proceso de predicción
if img is not None:
    size = (300,300)
    image = preproc_img(img)
    prediction = np.round(model.predict(image, verbose=0)[0][0] * 100, 2)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Probabilidad de cáncer: \n ### {prediction}%")
    with col2:
        st.image(img)
    with col3:
        st.image(main_funct(img, model))

# Formulario para los datos del paciente
with st.form('datos_paciente', clear_on_submit=True):
    st.subheader("Ingrese datos del paciente")
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.date_input('Edad', format='DD/MM/YYYY')
        trabajo = st.selectbox(label="Trabajo", options=categorias_trabajo, placeholder="Elija una opción")
        region = st.selectbox(label="Región Residencia", options=regiones_chile, placeholder="Ingrese opción" )
        ef_cron = st.selectbox(label="Enfermedad crónica", options=enfermedades_cronicas_comunes, placeholder="Ingrese Opción...")
    with col2:
        sexo = st.radio('Sexo', ['Masculino', 'Femenino'])
        canc_ant = st.radio('Antecedentes familiares de cáncer', ['Si', 'No'])
        ant_cancer = st.radio('Paciente con cáncer anteriormente', ['Si', 'No'])
        canc_diag = st.radio('Diagnóstico Radiólogo', ['Maligno', 'Benigno', 'Requiere más evaluación'])
    submit_button = st.form_submit_button('Enviar diagnóstico')

# Botón de submit
if submit_button:
    if prediction is None:
        st.error("Debes realizar una predicción con la IA")
    else:
        insert_patient()
        save_image(img)

    
