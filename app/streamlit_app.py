# ============================================================
# TOMATO LEAF DISEASE DETECTION
# STREAMLIT APPLICATION
# STAGE 8 + FASTAPI INTEGRATION + STAGE 10.1
# ============================================================

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

from datetime import datetime 

import pandas as pd
# ============================================================
# 8.1 — PROJECT ROOT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.inference.predict import load_class_names

from src.api.client import (
    predict_from_api,
    check_api_health
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)
# ============================================================
# 10.2 — PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []
    # ============================================================

# ============================================================
# 8.2 — PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .main-subtitle {
        text-align: center;
        color: #666666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-label {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .prediction-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 1rem;
    }

    .prediction-result-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #d9d9d9;
        background-color: #f8f9fa;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .prediction-label {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 0.2rem;
    }

    .prediction-disease {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .confidence-label {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 0.2rem;
    }

    .confidence-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8.3 — SIDEBAR PROJECT INFORMATION
# ============================================================

with st.sidebar:

    st.header("🍅 About Project")

    st.write(
        "An AI-powered application for detecting "
        "tomato leaf diseases from uploaded images."
    )

    st.divider()

    st.subheader("Model")

    st.write("Fine-Tuned MobileNetV2")
    st.write("Input Size: 224 × 224")
    st.write("Classes: 10")

    st.divider()

    st.subheader("Technology")

    st.write("🐍 Python")
    st.write("🧠 TensorFlow / Keras")
    st.write("🎨 Streamlit")
    st.write("🖼️ PIL")
    st.write("⚡ FastAPI")

    # --------------------------------------------------------
    # API STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔌 API Status")

    try:

        api_health = check_api_health()

        if api_health.get("status") == "healthy":

            st.success("🟢 FastAPI Connected")

        else:

            st.warning("🟡 FastAPI Running")

    except Exception:

        st.error("🔴 FastAPI Disconnected")


# ============================================================
# 8.4 — APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🍅 Tomato Leaf Disease Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Upload a tomato leaf image and the trained model '
    'will predict the most likely disease.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

class_names = load_class_names()


# ============================================================
# 8.5 — IMAGE UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-label">📤 Upload Tomato Leaf</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a clear image of a tomato leaf "
    "to get a disease prediction."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG."
)


# ============================================================
# 8.6 — IMAGE PREVIEW + VALIDATION
# ============================================================

image = None
image_valid = False

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        width, height = image.size

        col1, col2 = st.columns([2, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Tomato Leaf",
                width="stretch"
            )

        with col2:

            st.markdown("### 📐 Image Information")

            st.write(f"**Width:** {width}px")
            st.write(f"**Height:** {height}px")
            st.write(f"**Format:** {image.format}")

        # ----------------------------------------------------
        # IMAGE VALIDATION
        # ----------------------------------------------------

        if width < 100 or height < 100:

            st.error(
                "Image is too small. "
                "Please upload a clearer tomato leaf image."
            )

        else:

            image_valid = True

            st.success(
                "Image size is acceptable."
            )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded image: {e}"
        )


# ============================================================
# 8.7 — DISEASE PREDICTION
# ============================================================

if image_valid:

    st.divider()

    st.subheader("🔬 Disease Prediction")

    st.write(
        "Click the button below to analyze the uploaded image."
    )

    if st.button(
        "🔍 Predict Disease",
        width="stretch"
    ):

        with st.spinner("Analyzing image..."):

            try:

                uploaded_file.seek(0)

                result = predict_from_api(uploaded_file)

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                st.stop()


        # ====================================================
        # NORMALIZE FASTAPI RESPONSE
        # ====================================================

        predicted_class = result.get(
            "predicted_class",
            result.get("class")
        )

        confidence = result.get(
            "confidence",
            0
        )

        probabilities = result.get(
            "probabilities",
            []
        )
        # ============================================================
        # 10.2 — SAVE PREDICTION HISTORY
        # ============================================================

        prediction_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": uploaded_file.name,
            "prediction": predicted_class,
            "confidence": confidence
        }

        st.session_state.prediction_history.append(
            prediction_record
        )
        

        # ====================================================
        # VALIDATE API RESPONSE
        # ====================================================

        if predicted_class is None:

            st.error(
                "Prediction response did not contain "
                "a predicted class."
            )

            

            st.stop()


        if not probabilities:

            st.warning(
                "Prediction was returned, but probability "
                "information was not available."
            )


        # ====================================================
        # 10.1 — IMPROVED PREDICTION RESULT UI
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        

        # ----------------------------------------------------
        # Prediction metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Disease",
                predicted_class
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Classes Evaluated",
                len(class_names)
            )


        # ----------------------------------------------------
        # Confidence progress
        # ----------------------------------------------------

        st.write("### 📊 Prediction Confidence")

        st.progress(
            min(max(float(confidence), 0.0), 1.0)
        )

        st.caption(
            f"Model confidence: {confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Confidence interpretation
        # ----------------------------------------------------

        if confidence >= 0.80:

            st.success(
                "🟢 High confidence prediction. "
                "The model has strong confidence in this result."
            )

        elif confidence >= 0.60:

            st.warning(
                "🟡 Moderate confidence prediction. "
                "Consider uploading a clearer leaf image "
                "for better reliability."
            )

        else:

            st.error(
                "🔴 Low confidence prediction. "
                "Please upload a clear image of a tomato leaf."
            )


        # ====================================================
        # 8.9 — TOP 3 PREDICTIONS
        # ====================================================

        if probabilities:

            st.subheader("🏆 Top 3 Predictions")

            number_of_predictions = min(
                len(probabilities),
                len(class_names)
            )

            top_indices = sorted(
                range(number_of_predictions),
                key=lambda i: probabilities[i],
                reverse=True
            )[:3]


            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                class_name = class_names[index]

                probability = probabilities[index]

                st.write(
                    f"**{rank}. {class_name}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        # ====================================================
        # 8.10 — PREDICTION DETAILS
        # ====================================================

        with st.expander(
            "🔎 View Prediction Details"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Predicted class:** "
                    f"{predicted_class}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{confidence * 100:.2f}%"
                )

            with col2:

                st.write(
                    f"**Classes evaluated:** "
                    f"{len(class_names)}"
                )

                st.write(
                    "**Model:** Fine-Tuned MobileNetV2"
                )

            st.write(
                "The model evaluated all disease classes "
                "and returned the class with the highest "
                "predicted probability."
            )


# ============================================================
# 8.11 — ABOUT THE MODEL
# ============================================================

st.divider()

with st.expander(
    "ℹ️ About the Model"
):

    st.write(
        "This application uses a deep learning image "
        "classification model trained to recognize "
        "tomato leaf diseases."
    )

    st.write(
        "**Model:** Fine-Tuned MobileNetV2"
    )

    st.write(
        "**Input Image Size:** 224 × 224 pixels"
    )

    st.write(
        f"**Disease Classes:** {len(class_names)}"
    )

    st.write(
        "**Task:** Tomato Leaf Disease Classification"
    )

    st.write(
        "The model returns the predicted disease class "
        "along with its confidence and the top 3 predictions."
    )


# ============================================================
# NEW PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔄 New Prediction",
    width="stretch"
):

    st.rerun()

# ============================================================
# 10.3 — PREDICTION HISTORY UI
# ============================================================

st.divider()

st.subheader("📜 Prediction History")

if st.session_state.prediction_history:

    st.write(
        f"Total predictions: "
        f"**{len(st.session_state.prediction_history)}**"
    )

    for index, record in enumerate(
        reversed(st.session_state.prediction_history),
        start=1
    ):

        with st.expander(
            f"Prediction {index} — {record['prediction']}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Image:** {record['image']}"
                )

                st.write(
                    f"**Prediction:** "
                    f"{record['prediction']}"
                )

            with col2:

                st.write(
                    f"**Confidence:** "
                    f"{record['confidence'] * 100:.2f}%"
                )

                st.write(
                    f"**Time:** {record['time']}"
                )

else:

    st.info(
        "No prediction history yet. "
        "Upload an image and make a prediction."
    )

# ============================================================
# FOOTER
# ============================================================
# ============================================================
# 10.4 — CLEAR PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    if st.button(
        "🗑️ Clear Prediction History",
        width="stretch"
    ):

        st.session_state.prediction_history = []

        st.success(
            "Prediction history has been cleared."
        )

        st.rerun()


# ============================================================
# 10.5 — DOWNLOAD PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    csv_data = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        width="stretch"
    )

# ============================================================
# 10.6 — PREDICTION STATISTICS
# ============================================================

if st.session_state.prediction_history:

    st.subheader("📊 Prediction Statistics")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    total_predictions = len(history_df)

    average_confidence = (
        history_df["confidence"].mean()
    )

    highest_confidence = (
        history_df["confidence"].max()
    )

    lowest_confidence = (
        history_df["confidence"].min()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:

        st.metric(
            "Average Confidence",
            f"{average_confidence * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Highest Confidence",
            f"{highest_confidence * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Lowest Confidence",
            f"{lowest_confidence * 100:.2f}%"
        )


# ============================================================
# 10.7 — MOST FREQUENT PREDICTION
# ============================================================

if st.session_state.prediction_history:

    st.subheader("🏆 Most Frequently Predicted Disease")

    prediction_counts = (
        history_df["prediction"]
        .value_counts()
    )

    most_common_disease = (
        prediction_counts.index[0]
    )

    most_common_count = (
        prediction_counts.iloc[0]
    )

    st.success(
        f"🍅 **{most_common_disease}** "
        f"was predicted {most_common_count} time(s)."
    )

    st.write("### Disease Prediction Counts")

    for disease, count in prediction_counts.items():

        st.write(
            f"**{disease}** — {count} prediction(s)"
        )

        st.progress(
            int(
                count /
                prediction_counts.max() *
                100
            )
        )

# ============================================================
# 10.8 — PREDICTION HISTORY VISUALIZATION
# ============================================================

if st.session_state.prediction_history:

    st.subheader("📈 Prediction History Visualization")

    prediction_chart_data = (
        pd.DataFrame(
            st.session_state.prediction_history
        )["prediction"]
        .value_counts()
        .rename_axis("Disease")
        .reset_index(name="Predictions")
    )

    st.bar_chart(
        prediction_chart_data,
        x="Disease",
        y="Predictions"
    )


# ============================================================
# 10.9 — CONFIDENCE TREND VISUALIZATION
# ============================================================

if st.session_state.prediction_history:

    st.subheader("📉 Prediction Confidence Trend")

    confidence_df = pd.DataFrame(
        st.session_state.prediction_history
    ).copy()

    confidence_df["Prediction"] = range(
        1,
        len(confidence_df) + 1
    )

    confidence_df["Confidence (%)"] = (
        confidence_df["confidence"] * 100
    )

    st.line_chart(
        confidence_df,
        x="Prediction",
        y="Confidence (%)"
    )
st.markdown(
    """
    <div style="
        text-align:center;
        color:#888888;
        margin-top:2rem;
        font-size:0.85rem;
    ">
        🍅 Tomato Leaf Disease Detection |
        Fine-Tuned MobileNetV2 |
        Streamlit + FastAPI
    </div>
    """,
    unsafe_allow_html=True
)