import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_my_model():
    return load_model("brain_tumor_model.keras")

with st.spinner("Loading AI Model... Please wait."):
    model = load_my_model()

# ---------------- CLASS NAMES ---------------- #

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ---------------- UI ---------------- #

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload an MRI scan image and the model will predict the tumor type."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    key="mri_upload"
)

# ---------------- PREDICTION ---------------- #

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Scan",
        use_container_width=True
    )

    # Preprocessing
    image = image.resize((224, 224))

    img_array = img_to_array(image)

    # Agar model ke andar Rescaling(1./255) layer hai
    # to ye line comment hi rehne do
    # img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🧠 Analysing MRI Image..."):

        # Direct model call (predict se faster)
        prediction = model(
            img_array,
            training=False
        )

        prediction = prediction.numpy()

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        predicted_class = class_names[predicted_index]

    # ---------------- RESULTS ---------------- #

    st.success(
        f"Prediction: {predicted_class.upper()}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.subheader("Class Probabilities")

    for cls, prob in zip(class_names, prediction[0]):
        st.write(
            f"{cls}: {prob * 100:.2f}%"
        )
