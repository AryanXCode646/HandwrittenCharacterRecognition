"""Streamlit UI for Handwritten Digit Recognition (MNIST CNN)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from handwritten_recognition.predict import load_trained_model, predict_digit

st.set_page_config(page_title="Handwritten Digit Recognition — CodeAlpha", layout="centered")
st.title("Handwritten Digit Recognition (MNIST CNN)")
st.caption("Draw a digit (0–9) on the canvas or upload an image to classify it.")


@st.cache_resource
def _get_model():
    return load_trained_model("models/mnist_cnn.pt")


try:
    model = _get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tab_draw, tab_upload = st.tabs(["Draw Digit", "Upload Image"])

with tab_draw:
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=224,
        width=224,
        drawing_mode="freedraw",
        key="digit_canvas",
    )
    if st.button("Predict Drawn Digit") and canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype(np.uint8)
        rgb_img = Image.fromarray(rgba[:, :, :3])
        res = predict_digit(rgb_img, model, auto_invert=False)
        st.subheader(f"Predicted Digit: {res['predicted_digit']} ({res['confidence']:.1%})")
        prob_df = pd.DataFrame(
            {
                "Digit": list(res["probabilities"].keys()),
                "Probability": list(res["probabilities"].values()),
            }
        )
        st.bar_chart(prob_df.set_index("Digit"))

with tab_upload:
    uploaded = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=160)
        res = predict_digit(img, model, auto_invert=True)
        st.subheader(f"Predicted Digit: {res['predicted_digit']} ({res['confidence']:.1%})")
        prob_df = pd.DataFrame(
            {
                "Digit": list(res["probabilities"].keys()),
                "Probability": list(res["probabilities"].values()),
            }
        )
        st.bar_chart(prob_df.set_index("Digit"))
