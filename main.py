import os
import json
from io import BytesIO

from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ✅ IMPORT YOUR PREPROCESSING PIPELINE
from image_preprocessing import preprocess_image



# PAGE CONFIG (NO SIDEBAR)

st.set_page_config(
    page_title="Tomato Disease Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================
# FULL SCREEN BACKGROUND + NAV BUTTON STYLE
# ======================================================
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background-image: url("https://thumbs.dreamstime.com/b/tomato-plant-disease-prevention-tips-garden-infographic-outdoor-close-up-health-awareness-learn-how-to-identify-prevent-367960390.jpg");
    background-size: cover;
    background-position: center;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stAppViewContainer"] > .main {
    background: rgba(255, 255, 255, 0.88);
    min-height: 100vh;
    padding: 2rem;
}

/* Floating navigation button */
.nav-btn {
    position: fixed;
    bottom: 25px;
    right: 25px;
    z-index: 9999;
}

.nav-btn button {
    background-color: #0B5ED7;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 30px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
}

.nav-btn button:hover {
    background-color: #084298;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# CONSTANTS
# ======================================================
CONFIDENCE_THRESHOLD = 0.75


# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model (1).keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")


# ======================================================
# LOAD MODEL & CLASS INDICES
# ======================================================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_class_indices(path):
    with open(path, "r") as f:
        data = json.load(f)

    class_map = {}
    if all(isinstance(v, int) for v in data.values()):
        for name, idx in data.items():
            class_map[idx] = name
    else:
        for idx, name in data.items():
            class_map[int(idx)] = name

    return class_map


model = load_model(MODEL_PATH)
class_indices = load_class_indices(CLASS_INDEX_PATH)


# ======================================================
# DISEASE PREVENTION DATA
# ======================================================
DISEASE_PREVENTION = {
    "Tomato_Bacterial_spot": [
        "Remove infected leaves immediately.",
        "Avoid overhead watering.",
        "Use copper-based bactericide."
    ],
    "Tomato_Early_blight": [
        "Rotate crops every season.",
        "Remove affected leaves.",
        "Apply copper fungicide."
    ],
    "Tomato_Late_blight": [
        "Destroy infected plants.",
        "Use disease-free seedlings."
    ],
    "Tomato_Leaf_mold": [
        "Improve air circulation.",
        "Reduce humidity."
    ],
    "Tomato_mosaic_virus": [
        "Use resistant varieties.",
        "Disinfect tools regularly."
    ],
    "Tomato_healthy": [
        "Maintain balanced fertilization.",
        "Ensure proper spacing."
    ]
}

GENERAL_TIPS = [
    "Water plants at soil level.",
    "Remove plant debris regularly.",
    "Monitor plants frequently."
]


# ======================================================
# SESSION STATE
# ======================================================
st.session_state.setdefault("section", "Classifier")
st.session_state.setdefault("last_predicted", None)
st.session_state.setdefault("cart", [])


# ======================================================
# CLASSIFIER PAGE
# ======================================================
if st.session_state["section"] == "Classifier":

    st.title("🌱 Tomato Disease Classifier")
    st.write("Upload a tomato leaf image to identify disease.")

    uploaded_file = st.file_uploader(
        "Upload image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        st.image(image, width=350)

        if st.button("Classify"):
            batch = preprocess_image(image)
            preds = model.predict(batch)[0]

            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            label = class_indices[idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                st.success(
                    f"Prediction: **{label}** ({confidence*100:.2f}% confidence)"
                )

                st.subheader("🌿 Disease Prevention Tips")
                for tip in DISEASE_PREVENTION.get(label, []):
                    st.write("✅", tip)

                st.subheader("🌱 General Care Tips")
                for tip in GENERAL_TIPS:
                    st.write("•", tip)

                st.session_state["last_predicted"] = label

            else:
                st.warning(
                    "Low confidence prediction. "
                    "Please upload a clearer leaf image."
                )

    # 👉 CLEAR NAVIGATION BUTTON
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("➡ Go to Store", key="to_store"):
        st.session_state["section"] = "Store"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# STORE PAGE
# ======================================================
else:

    st.title("🛒 Store — Disease Prevention Materials")

    DISEASE_STORE = {
        "Tomato_Early_blight": [
            ("Copper Fungicide", "https://www.amazon.in/s?k=copper+fungicide"),
            ("Neem Oil", "https://www.amazon.in/s?k=neem+oil")
        ],
        "Tomato_Late_blight": [
            ("Systemic Fungicide", "https://www.amazon.in/s?k=systemic+fungicide")
        ],
        "Tomato_mosaic_virus": [
            ("Insecticidal Soap", "https://www.amazon.in/s?k=insecticidal+soap"),
            ("70% Isopropyl Alcohol", "https://www.amazon.in/s?k=70%25+isopropyl+alcohol")
        ]
    }

    GENERAL_STORE = [
        ("Pruning Shears", "https://www.amazon.in/s?k=pruning+shears"),
        ("Garden Gloves", "https://www.amazon.in/s?k=garden+gloves"),
        ("Soil pH Meter", "https://www.amazon.in/s?k=soil+ph+meter")
    ]

    label = st.session_state.get("last_predicted")
    if label:
        st.info(f"Detected disease: **{label}**")
    else:
        st.info("No disease detected yet.")

    def product_selector(items, prefix):
        selected = []
        for i, (name, link) in enumerate(items):
            if st.checkbox(name, key=f"{prefix}_{i}"):
                selected.append((name, link))
        return selected

    selected_items = []

    if label in DISEASE_STORE:
        st.subheader("Recommended Products")
        selected_items += product_selector(DISEASE_STORE[label], "disease")

    st.subheader("General Gardening Items")
    selected_items += product_selector(GENERAL_STORE, "general")

    if st.button("Open Selected Links"):
        for name, link in selected_items:
            st.markdown(f"- [{name}]({link})")

    # 👉 CLEAR BACK NAVIGATION
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("⬅ Back to Classifier", key="back_classifier"):
        st.session_state["section"] = "Classifier"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

