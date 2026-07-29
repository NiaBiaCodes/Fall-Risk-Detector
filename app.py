from pathlib import Path
import pickle
import traceback
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Wearable Fall Risk Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Temporary diagnostics: confirms which app file and model are being used.
# Remove these two lines after the model loads successfully if desired.
st.info(f"Running app from: {Path(__file__).resolve()}")
st.info("Selected model: cstick_model.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-container {
        padding: 4rem 3rem;
        margin-bottom: 2.5rem;
        border: 1px solid #2f3542;
        border-radius: 22px;
        background:
            radial-gradient(
                circle at top right,
                rgba(63, 187, 135, 0.18),
                transparent 35%
            ),
            linear-gradient(135deg, #161b22, #101419);
    }

    .eyebrow {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(72, 199, 142, 0.45);
        border-radius: 999px;
        background-color: rgba(72, 199, 142, 0.10);
        color: #72d6aa;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04rem;
        text-transform: uppercase;
    }

    .hero-title {
        max-width: 800px;
        margin: 0;
        color: #ffffff;
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1.08;
    }

    .hero-description {
        max-width: 800px;
        margin-top: 1.35rem;
        color: #c4c9d2;
        font-size: 1.13rem;
        line-height: 1.75;
    }

    .hero-note {
        margin-top: 1.5rem;
        color: #8f98a8;
        font-size: 0.95rem;
    }

    .section-label {
        color: #72d6aa;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
    }

    .section-title {
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 750;
    }

    .section-description {
        margin-bottom: 1.5rem;
        color: #aeb5c1;
        line-height: 1.65;
    }

    .info-card {
        height: 100%;
        min-height: 175px;
        padding: 1.5rem;
        border: 1px solid #303640;
        border-radius: 16px;
        background-color: #171b22;
    }

    .info-card h3 {
        margin: 0 0 0.6rem 0;
        color: #ffffff;
        font-size: 1rem;
    }

    .info-card p {
        margin: 0;
        color: #aeb5c1;
        line-height: 1.6;
    }

    .reference-box {
        padding: 0.85rem 1rem;
        margin-top: 0.45rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4d8df7;
        border-radius: 8px;
        background-color: rgba(77, 141, 247, 0.09);
        color: #bac6d8;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .risk-card {
        padding: 2rem;
        margin-top: 1.5rem;
        border-radius: 18px;
    }

    .risk-low {
        border: 1px solid rgba(46, 204, 113, 0.55);
        background-color: rgba(46, 204, 113, 0.11);
    }

    .risk-moderate {
        border: 1px solid rgba(243, 156, 18, 0.60);
        background-color: rgba(243, 156, 18, 0.12);
    }

    .risk-high {
        border: 1px solid rgba(231, 76, 60, 0.60);
        background-color: rgba(231, 76, 60, 0.12);
    }

    .risk-heading {
        margin-bottom: 0.5rem;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 650;
    }

    .risk-value {
        margin: 0;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
    }

    .risk-summary {
        margin-top: 0.8rem;
        color: #d0d5dd;
        line-height: 1.6;
    }

    .factor {
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #586174;
        border-radius: 8px;
        background-color: #181d25;
    }

    .factor-title {
        margin-bottom: 0.25rem;
        color: #ffffff;
        font-weight: 700;
    }

    .factor-description {
        margin: 0;
        color: #abb3c0;
        line-height: 1.5;
    }

    .disclaimer {
        padding: 1.25rem 1.4rem;
        margin-top: 2rem;
        border: 1px solid #343b47;
        border-radius: 14px;
        background-color: #151920;
        color: #9fa7b5;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 3.2rem;
        border: none;
        border-radius: 10px;
        background-color: #2fb579;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        border: none;
        background-color: #279d69;
        color: #ffffff;
    }

    [data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid #303640;
        border-radius: 12px;
        background-color: #171b22;
    }

    hr {
        margin-top: 3rem;
        margin-bottom: 3rem;
        border-color: #292f39;
    }

    @media (max-width: 700px) {
        .hero-container {
            padding: 2.5rem 1.5rem;
        }

        .hero-title {
            font-size: 2.35rem;
        }
    }
</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL FILE SETTINGS
# =========================================================

# Add your exact filename here when you know it.
# Examples:
# MODEL_FILENAME = "fall_risk_model.pkl"
# MODEL_FILENAME = "random_forest_model.joblib"
#
# Leave it as None to let the app search automatically.
MODEL_FILENAME = "cstick_model.pkl"

# Folders the app will search.
MODEL_FOLDERS = [
    Path("."),
    Path("models"),
    Path("model"),
    Path("saved_models"),
]

# Supported saved-model file extensions.
MODEL_EXTENSIONS = [
    "*.pkl",
    "*.pickle",
    "*.joblib",
]


# =========================================================
# MODEL LOADING FUNCTIONS
# =========================================================
def find_model_file():
    """
    Finds the model file.

    Priority:
    1. The exact filename entered in MODEL_FILENAME
    2. A supported model file in the project folders
    """

    if MODEL_FILENAME:
        possible_paths = [
            Path(MODEL_FILENAME),
            Path("models") / MODEL_FILENAME,
            Path("model") / MODEL_FILENAME,
            Path("saved_models") / MODEL_FILENAME,
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                return path

    discovered_models = []

    for folder in MODEL_FOLDERS:
        if not folder.exists():
            continue

        for extension in MODEL_EXTENSIONS:
            discovered_models.extend(folder.glob(extension))

    # Remove duplicate paths.
    discovered_models = list(dict.fromkeys(discovered_models))

    if len(discovered_models) == 1:
        return discovered_models[0]

    if len(discovered_models) > 1:
        # Prefer filenames containing relevant words.
        preferred_words = [
            "fall",
            "risk",
            "model",
            "classifier",
            "random_forest",
        ]

        for word in preferred_words:
            for path in discovered_models:
                if word in path.name.lower():
                    return path

        return discovered_models[0]

    return None


@st.cache_resource
def load_model(model_path_string):
    model_path = Path(model_path_string)

    with open(model_path, "rb") as model_file:
        model_bundle = pickle.load(model_file)

    if not isinstance(model_bundle, dict):
        raise TypeError(
            "The saved model file must contain a dictionary."
        )

    required_keys = {
        "model",
        "scaler",
        "features",
        "class_names",
    }

    missing_keys = required_keys - model_bundle.keys()

    if missing_keys:
        raise KeyError(
            "Missing model information: "
            + ", ".join(sorted(missing_keys))
        )

    return model_bundle

def list_project_files():
    """
    Returns visible project files to help diagnose model-path problems.
    """

    visible_files = []

    for path in Path(".").rglob("*"):
        if (
            path.is_file()
            and ".git" not in path.parts
            and ".venv" not in path.parts
            and "__pycache__" not in path.parts
        ):
            visible_files.append(str(path))

    return sorted(visible_files)

model_path = find_model_file()

model_bundle = None
model = None
scaler = None
model_features = None
class_names = None

model_loading_error = None
model_loading_traceback = None

if model_path is not None:
    try:
        model_bundle = load_model(str(model_path))

        model = model_bundle["model"]
        scaler = model_bundle["scaler"]
        model_features = model_bundle["features"]
        class_names = model_bundle["class_names"]

    except Exception as error:
        model_loading_error = f"{type(error).__name__}: {repr(error)}"
        model_loading_traceback = traceback.format_exc()


# =========================================================
# PREDICTION HELPERS
# =========================================================
def normalize_prediction(prediction):
    """
    Converts the model's class label into low, moderate, or high.

    Change these mappings if your model uses different labels.
    """

    prediction_text = str(prediction).strip().lower()

    low_labels = {
        "0",
        "low",
        "none",
        "no risk",
        "normal",
        "adl",
        "low risk",
        "no fall",
    }

    moderate_labels = {
        "1",
        "moderate",
        "medium",
        "maybe",
        "near fall",
        "near-fall",
        "possible",
        "possible risk",
        "moderate risk",
    }

    high_labels = {
        "2",
        "high",
        "fall",
        "high risk",
        "severe",
    }

    if prediction_text in low_labels:
        return "low"

    if prediction_text in moderate_labels:
        return "moderate"

    if prediction_text in high_labels:
        return "high"

    return "moderate"


def get_risk_information(risk_level):
    risk_information = {
        "low": {
            "label": "Low Detectable Risk",
            "symbol": "●",
            "css_class": "risk-low",
            "summary": (
                "The model did not detect a strong pattern associated with "
                "elevated fall risk in the submitted measurements."
            ),
        },
        "moderate": {
            "label": "Possible Fall Risk",
            "symbol": "●",
            "css_class": "risk-moderate",
            "summary": (
                "The model detected some measurements that may be associated "
                "with reduced stability or increased fall susceptibility."
            ),
        },
        "high": {
            "label": "High Detectable Risk",
            "symbol": "●",
            "css_class": "risk-high",
            "summary": (
                "The model detected a pattern in the submitted measurements "
                "that is more strongly associated with fall events."
            ),
        },
    }

    return risk_information[risk_level]


def get_prediction_confidence(trained_model, input_data):
    """
    Gets the highest class probability when predict_proba is supported.
    """

    if not hasattr(trained_model, "predict_proba"):
        return None

    probabilities = trained_model.predict_proba(input_data)[0]

    return float(np.max(probabilities))


def explain_measurements(
    movement_distance,
    hrv,
    glucose_level,
    oxygen_level,
):
    """
    Generates plain-language explanations.

    The thresholds below are explanatory interface values.
    They may not be the exact thresholds learned by the model.
    """

    factors = []

    # Oxygen
    if oxygen_level < 90:
        factors.append(
            {
                "title": "Very low submitted oxygen saturation",
                "description": (
                    "A low SpO₂ reading may be associated with symptoms such as "
                    "weakness, fatigue, confusion, or dizziness. These symptoms "
                    "could make safe movement more difficult."
                ),
            }
        )
    elif oxygen_level < 95:
        factors.append(
            {
                "title": "Oxygen saturation is below the general reference range",
                "description": (
                    "The entered SpO₂ value is below the commonly cited range of "
                    "95% to 100%. Personal medical conditions, elevation, and "
                    "device accuracy can affect what is appropriate."
                ),
            }
        )
    else:
        factors.append(
            {
                "title": "Oxygen saturation is not being flagged",
                "description": (
                    "The submitted SpO₂ value is within the application's general "
                    "reference range."
                ),
            }
        )

    # Movement
    # Replace these thresholds with values appropriate for your dataset.
    if movement_distance < 1:
        factors.append(
            {
                "title": "Very limited submitted movement distance",
                "description": (
                    "A low movement value may reflect reduced mobility, limited "
                    "activity, cautious movement, or difficulty completing normal "
                    "movement."
                ),
            }
        )
    elif movement_distance < 3:
        factors.append(
            {
                "title": "Reduced submitted movement distance",
                "description": (
                    "Reduced movement may be associated with lower mobility or "
                    "possible difficulty maintaining stable movement."
                ),
            }
        )
    else:
        factors.append(
            {
                "title": "Movement distance is not being flagged",
                "description": (
                    "The movement value does not fall below the application's "
                    "current explanatory threshold."
                ),
            }
        )

    # HRV
    # These are example research-interface ranges only.
    if hrv < 20:
        factors.append(
            {
                "title": "Low submitted HRV value",
                "description": (
                    "The HRV value falls within the application's lower explanatory "
                    "range. HRV must be interpreted using the specific wearable, "
                    "measurement method, age, and personal baseline."
                ),
            }
        )
    elif hrv < 40:
        factors.append(
            {
                "title": "HRV may require additional context",
                "description": (
                    "The HRV value is within the application's caution range. "
                    "There is no single healthy HRV value that applies to everyone."
                ),
            }
        )
    else:
        factors.append(
            {
                "title": "HRV is not being flagged",
                "description": (
                    "The HRV value does not fall below the application's current "
                    "explanatory threshold."
                ),
            }
        )

    # Blood glucose
    if glucose_level < 70:
        factors.append(
            {
                "title": "Low submitted blood glucose",
                "description": (
                    "Low blood glucose can be associated with shakiness, weakness, "
                    "confusion, or dizziness, which may affect safe movement."
                ),
            }
        )
    elif glucose_level > 180:
        factors.append(
            {
                "title": "Elevated submitted blood glucose",
                "description": (
                    "The entered glucose value is above the application's general "
                    "post-meal reference. Interpretation depends on when the "
                    "measurement was taken and the person's medical guidance."
                ),
            }
        )
    else:
        factors.append(
            {
                "title": "Blood glucose is not being strongly flagged",
                "description": (
                    "The submitted glucose value does not fall outside the broad "
                    "range used by this explanation system. Meal timing still "
                    "affects interpretation."
                ),
            }
        )

    return factors


def get_precaution_message(risk_level):
    messages = {
        "low": (
            "The submitted measurements do not show a strong detectable fall-risk "
            "pattern. Continue monitoring for meaningful changes in movement, "
            "balance, weakness, or dizziness."
        ),
        "moderate": (
            "Some measurements may affect mobility or balance. The person may "
            "benefit from moving carefully, using stable support when necessary, "
            "and discussing persistent dizziness, weakness, or instability with "
            "a qualified healthcare professional."
        ),
        "high": (
            "The submitted measurements contain a stronger fall-risk pattern. "
            "The person should avoid moving without assistance when feeling dizzy, "
            "weak, or unstable and should seek evaluation from a qualified "
            "healthcare professional."
        ),
    }

    return messages[risk_level]


def build_input_dataframe(
    movement_distance,
    hrv,
    glucose_level,
    oxygen_level,
    feature_names,
):
    """
    Builds the input DataFrame using the exact feature names and order
    stored inside the saved model bundle.
    """

    entered_values = {
        "Distance": movement_distance,
        "HRV": hrv,
        "Sugar level": glucose_level,
        "SpO2": oxygen_level,
    }

    missing_features = [
        feature
        for feature in feature_names
        if feature not in entered_values
    ]

    if missing_features:
        raise ValueError(
            "The model expects features that are not collected by the form: "
            + ", ".join(str(feature) for feature in missing_features)
        )

    ordered_values = {
        feature: entered_values[feature]
        for feature in feature_names
    }

    return pd.DataFrame([ordered_values])

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(
    """
<div class="hero-container">
    <div class="eyebrow">AI4ALL Research Project</div>

<h1 class="hero-title">
        Wearable Fall Risk Detection
</h1>

<p class="hero-description">
        This project explores whether wearable sensor measurements can help
        distinguish normal movement from possible near-fall and fall events.
        Enter movement and physiological measurements below to receive a
        machine-learning risk estimate and an explanation of the factors
        that may require additional attention.
</p>

<p class="hero-note">
        Machine learning • Wearable sensors • Mobility research
</p>
</div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT CARDS
# =========================================================
card_one, card_two, card_three = st.columns(3)

with card_one:
    st.markdown(
        """
<div class="info-card">
    <h3>Wearable measurements</h3>
    <p>
        The assessment uses movement distance, heart rate variability,
        blood glucose, and blood oxygen saturation.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

with card_two:
    st.markdown(
        """
<div class="info-card">
    <h3>Research purpose</h3>
    <p>
        The project investigates whether changes in wearable data can
        support earlier recognition of mobility concerns.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

with card_three:
    st.markdown(
        """
<div class="info-card">
    <h3>Interpretable results</h3>
<p>
        The application presents a risk category and explains which
        submitted measurements may deserve attention.
</p>
</div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<hr>", unsafe_allow_html=True)


# =========================================================
# ASSESSMENT FORM
# =========================================================
st.markdown(
    """
<div class="section-label">Fall-risk assessment</div>
<div class="section-title">Enter wearable sensor measurements</div>
<div class="section-description">
    Select the question-mark icon beside each label to learn what the
    measurement means and see general reference information.
</div>
    """,
    unsafe_allow_html=True,
)


with st.form("fall_risk_form"):
    left_column, right_column = st.columns(2)

    with left_column:
        movement_distance = st.number_input(
            "Movement Distance",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "The amount of movement detected during the measurement period. "
                "Use the same unit and measurement period as the dataset used to "
                "train the model. There is no universal reference range because "
                "wearables can report movement in different ways."
            ),
        )

        st.markdown(
            """
<div class="reference-box">
    <strong>Reference:</strong> No universal range. The appropriate
    value depends on the wearable, unit, and recording period.
</div>
            """,
            unsafe_allow_html=True,
        )

        hrv = st.number_input(
            "Heart Rate Variability (HRV)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "HRV measures differences in time between consecutive heartbeats. "
                "It is not the same as heart rate. HRV may be measured using RMSSD, "
                "SDNN, or another method, commonly in milliseconds. There is no "
                "single appropriate HRV value for everyone. Enter the value and "
                "metric produced by the wearable used in this research."
            ),
        )

        st.markdown(
            """
<div class="reference-box">
    <strong>Reference:</strong> HRV should be compared with the
    person's usual baseline and the same device and HRV metric.
</div>
            """,
            unsafe_allow_html=True,
        )

    with right_column:
        glucose_level = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            format="%.1f",
            help=(
                "Blood glucose describes the amount of glucose in the blood. "
                "Typical targets for many people with diabetes are 80–130 mg/dL "
                "before a meal and below 180 mg/dL one to two hours after the "
                "start of a meal. Personal targets may differ."
            ),
        )

        st.markdown(
            """
<div class="reference-box">
    <strong>General context:</strong> 80–130 mg/dL before a meal;
    below 180 mg/dL about 1–2 hours after eating. Personal targets vary.
</div>
            """,
            unsafe_allow_html=True,
        )

        oxygen_level = st.number_input(
            "Oxygen Saturation (SpO₂ %)",
            min_value=0.0,
            max_value=100.0,
            value=98.0,
            step=0.1,
            format="%.1f",
            help=(
                "SpO₂ estimates the percentage of red blood cells carrying oxygen. "
                "It can be measured with a pulse oximeter or supported wearable. "
                "A commonly cited normal range is approximately 95%–100%, though "
                "lung conditions, elevation, circulation, skin temperature, and "
                "device accuracy may affect readings."
            ),
        )

        st.markdown(
            """
<div class="reference-box">
    <strong>General reference:</strong> Approximately 95%–100%.
    Medical conditions and elevation may affect an appropriate reading.
</div>
            """,
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button(
        "Predict Fall Risk",
        use_container_width=True,
    )


# =========================================================
# PREDICTION RESULTS
# =========================================================
if submitted:
    if model_path is None:
        st.error(
            "No saved machine-learning model was found in the application files."
        )

        st.markdown(
            """
            ### How to fix this

            Your saved model must be committed to the GitHub repository used by
            Streamlit. Place it in either the main folder or a folder named
            `models`.

            Example project structure:

            ```text
            ai4all/
            ├── app.py
            ├── requirements.txt
            ├── fall_risk_model.pkl
            └── README.md
            ```

            Or:

            ```text
            ai4all/
            ├── app.py
            ├── requirements.txt
            ├── models/
            │   └── fall_risk_model.pkl
            └── README.md
            ```
            """
        )

        project_files = list_project_files()

        with st.expander("View files currently available to the app"):
            if project_files:
                st.code("\n".join(project_files))
            else:
                st.write("No project files were detected.")

        st.stop()

    if model_loading_error:
        st.error(
            f"The model file `{model_path}` was found, but it could not be loaded."
        )

        st.write("Error type and message:")
        st.code(model_loading_error)

        with st.expander("View complete traceback"):
            st.code(model_loading_traceback or "No traceback was captured.")

        st.info(
            "This usually means the model file is incomplete, was saved with a "
            "missing dependency, or was created using incompatible package versions."
        )

        st.stop()

    input_data = build_input_dataframe(
        movement_distance=movement_distance,
        hrv=hrv,
        glucose_level=glucose_level,
        oxygen_level=oxygen_level,
        feature_names=model_features,

    )

    try:
        # Logistic Regression uses the saved scaler. A Decision Tree does not.
        if scaler is not None:
            prediction_input = scaler.transform(input_data)
        else:
            prediction_input = input_data

        raw_prediction = model.predict(prediction_input)[0]

        risk_level = normalize_prediction(raw_prediction)
        risk_information = get_risk_information(risk_level)

        confidence = get_prediction_confidence(
            trained_model=model,
            input_data=prediction_input,
        )

    except ValueError as error:
        st.error(
            "The model was loaded, but the input columns do not match the "
            "features used during model training."
        )

        st.code(str(error))

        if hasattr(model, "feature_names_in_"):
            st.write("The model expects these feature names:")

            st.code(
                "\n".join(
                    str(feature)
                    for feature in model.feature_names_in_
                )
            )

        st.write("The application currently sends these feature names:")

        st.code("\n".join(input_data.columns))

        st.stop()

    except Exception as error:
        st.error("An unexpected prediction error occurred.")
        st.code(str(error))
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="section-label">Assessment result</div>
<div class="section-title">Fall-risk detectability</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="risk-card {risk_information["css_class"]}">
    <div class="risk-heading">Prediction Result</div>

<p class="risk-value">
        {risk_information["symbol"]}
        {risk_information["label"]}
</p>

<p class="risk-summary">
        {risk_information["summary"]}
</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two = st.columns(2)

    with metric_one:
        st.metric(
            label="Risk category",
            value=risk_information["label"],
        )

    with metric_two:
        if confidence is not None:
            st.metric(
                label="Model confidence",
                value=f"{confidence * 100:.1f}%",
            )
        else:
            st.metric(
                label="Model confidence",
                value="Not available",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="section-label">Result interpretation</div>
<div class="section-title">
    Why might this person need to be careful?
</div>
<div class="section-description">
    These explanations describe possible concerns associated with the
    entered measurements. They do not prove that a measurement caused
    the model's prediction.
</div>
        """,
        unsafe_allow_html=True,
    )

    factors = explain_measurements(
        movement_distance=movement_distance,
        hrv=hrv,
        glucose_level=glucose_level,
        oxygen_level=oxygen_level,
    )

    if factors:
        for factor in factors:
            st.markdown(
                f"""
<div class="factor">
    <div class="factor-title">{factor["title"]}</div>
    <p class="factor-description">{factor["description"]}</p>
</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "No individual measurement crossed the application's explanatory "
            "thresholds. The prediction may reflect the combined pattern across "
            "multiple measurements rather than one value alone."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="section-label">Suggested caution</div>
<div class="section-title">
    What should the person consider?
</div>
        """,
        unsafe_allow_html=True,
    )

    st.info(get_precaution_message(risk_level))

    with st.expander("View submitted measurements"):
        submitted_values = pd.DataFrame(
            {
                "Measurement": [
                    "Movement Distance",
                    "Heart Rate Variability",
                    "Blood Glucose",
                    "Oxygen Saturation",
                ],
                "Submitted Value": [
                    movement_distance,
                    hrv,
                    glucose_level,
                    oxygen_level,
                ],
            }
        )

        st.dataframe(
            submitted_values,
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# MODEL STATUS
# =========================================================
with st.expander("Model status"):
    if model_path is None:
        st.warning("No supported model file was detected.")
    elif model_loading_error:
        st.error(f"Model detected but could not be loaded: {model_path}")
    else:
        st.success(f"Model loaded successfully: {model_path}")


# =========================================================
# DISCLAIMER
# =========================================================
st.markdown(
    """
<div class="disclaimer">
    <strong>Research disclaimer:</strong>
    This application is intended for educational and research purposes.
    It is not a medical device and does not provide a diagnosis, treatment
    recommendation, or emergency assessment. The displayed reference
    information is general and may not represent the thresholds learned by
    the machine-learning model. Medical concerns should be discussed with a
    qualified healthcare professional.
</div>
    """,
    unsafe_allow_html=True,
)