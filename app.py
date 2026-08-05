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


# =========================================================
# CUSTOM CSS — ORIGINAL DARK STYLE + VISIBILITY FIXES
# =========================================================
st.markdown(
    """
    <style>
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            max-width: 100%;
            overflow-x: hidden !important;
        }

        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        .block-container {
            width: 100%;
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 4rem;
            overflow-x: hidden;
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

        [data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

[data-testid="stColumn"] {
    display: flex;
}

[data-testid="stColumn"] > div {
    width: 100%;
}

.info-card {
    width: 100%;
    height: 100%;
    min-height: 245px;
    padding: 1.5rem;
    border: 1px solid #303640;
    border-radius: 16px;
    background-color: #171b22;
    box-sizing: border-box;
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

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        /* Keep tabs visible and preserve dark styling */
        [data-testid="stTabs"] {
            max-width: 100%;
            overflow-x: hidden;
        }

        [data-testid="stTabs"] [role="tablist"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        button[data-baseweb="tab"] {
            color: #aeb5c1 !important;
            font-weight: 650;
        }

        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            color: inherit !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #72d6aa !important;
        }

        /* Keep form text readable */
        [data-testid="stNumberInput"] label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
        }

        [data-testid="stNumberInput"] input {
            color: #ffffff !important;
            background-color: #262a33 !important;
        }

        [data-testid="stNumberInput"] button {
            color: #ffffff !important;
            background-color: #262a33 !important;
        }

        [data-testid="stNumberInput"] svg {
            fill: #ffffff !important;
        }

        /* Rectangular dropdown / expander styling */
[data-testid="stExpander"] {
    border: 1px solid #343b47 !important;
    border-radius: 14px !important;
    background-color: #151920 !important;
    overflow: hidden;
}

/* Closed and open expander header */
[data-testid="stExpander"] summary {
    background-color: #151920 !important;
    color: #ffffff !important;
}

/* Keep header dark when expanded */
[data-testid="stExpander"] details[open] > summary {
    background-color: #151920 !important;
    color: #ffffff !important;
    border-bottom: 1px solid #343b47 !important;
}

/* Header text and arrow */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Expanded content area */
[data-testid="stExpanderDetails"] {
    background-color: #151920 !important;
    color: #aeb5c1 !important;
}

[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] li {
    color: #aeb5c1 !important;
}

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #ffffff !important;
            font-weight: 650;
        }

        [data-testid="stExpanderDetails"],
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] span {
            color: #aeb5c1 !important;
        }

        /* Alert boxes */
        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Dataframes should scroll internally instead of shifting the page */
        [data-testid="stDataFrame"] {
            max-width: 100%;
            overflow-x: auto;
        }

        .graph-viewer-panel,
        .graph-nav-panel {
            padding: 1.2rem 1.25rem;
            border: 1px solid #303640;
            border-radius: 16px;
            background-color: #171b22;
            box-sizing: border-box;
        }

        .graph-nav-panel {
            position: sticky;
            top: 1.25rem;
        }

        .graph-nav-title {
            margin: 0 0 0.35rem 0;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 750;
        }

        .graph-nav-caption {
            margin-bottom: 0.9rem;
            color: #aeb5c1;
            line-height: 1.55;
            font-size: 0.93rem;
        }

        div[data-testid="stRadio"] {
            gap: 0.25rem;
        }

        div[data-testid="stRadio"] label {
            padding: 0.35rem 0.5rem 0.35rem 0.15rem;
            border-radius: 10px;
        }

        div[data-testid="stRadio"] label:hover {
            background-color: rgba(114, 214, 170, 0.08);
        }

        div[data-testid="stRadio"] label[data-checked="true"] {
            background-color: rgba(114, 214, 170, 0.14);
            border: 1px solid rgba(114, 214, 170, 0.45);
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

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .info-card {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL SETTINGS
# =========================================================
MODEL_FILENAME = "cstick_model.pkl"

MODEL_FOLDERS = [
    Path("."),
    Path("models"),
    Path("model"),
    Path("saved_models"),
]

MODEL_EXTENSIONS = [
    "*.pkl",
    "*.pickle",
    "*.joblib",
]

APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"


# =========================================================
# MODEL LOADING
# =========================================================
def find_model_file():
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

    discovered_models = list(dict.fromkeys(discovered_models))

    if not discovered_models:
        return None

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


@st.cache_resource
def load_model(model_path_string):
    model_path = Path(model_path_string)

    with open(model_path, "rb") as model_file:
        model_bundle = pickle.load(model_file)

    if not isinstance(model_bundle, dict):
        raise TypeError("The saved model file must contain a dictionary.")

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
    if not hasattr(trained_model, "predict_proba"):
        return None

    probabilities = trained_model.predict_proba(input_data)[0]
    return float(np.max(probabilities))


def get_confidence_explanation(confidence):
    if confidence is None:
        return (
            "This model does not provide class probabilities, so a confidence "
            "percentage cannot be displayed."
        )

    percentage = confidence * 100

    if percentage < 55:
        return (
            "The model confidence is low because more than one risk category "
            "received a similar probability. The submitted measurements may be "
            "close to a decision boundary, may contain a mixed pattern, or may "
            "not closely resemble one clear pattern in the training data."
        )

    if percentage < 70:
        return (
            "The model confidence is moderate. One category received the highest "
            "probability, but at least one other category remained reasonably "
            "plausible. Small changes in the measurements could potentially "
            "change the result."
        )

    return (
        "The model confidence is relatively strong because one category received "
        "a clearly higher probability than the alternatives. This does not mean "
        "the prediction is medically certain."
    )


def get_default_values(feature_names, fitted_scaler):
    """
    Uses training-set averages stored in the fitted scaler when available.
    Otherwise, it uses reasonable demonstration values.

    These values are examples only and are not clinical recommendations.
    """
    defaults = {
        "Distance": 3.0,
        "HRV": 40.0,
        "Sugar level": 100.0,
        "SpO2": 98.0,
    }

    if (
        fitted_scaler is not None
        and hasattr(fitted_scaler, "mean_")
        and feature_names is not None
        and len(fitted_scaler.mean_) == len(feature_names)
    ):
        for feature, mean_value in zip(feature_names, fitted_scaler.mean_):
            if feature in defaults and np.isfinite(mean_value):
                defaults[feature] = float(mean_value)

    defaults["Distance"] = max(0.0, defaults["Distance"])
    defaults["HRV"] = max(0.0, defaults["HRV"])
    defaults["Sugar level"] = max(0.0, defaults["Sugar level"])
    defaults["SpO2"] = min(100.0, max(0.0, defaults["SpO2"]))

    return defaults


def explain_measurements(
    movement_distance,
    hrv,
    glucose_level,
    oxygen_level,
):
    """
    Creates plain-language explanations based on interface thresholds.

    These thresholds are not exact model feature-attribution values.
    """
    factors = []

    if oxygen_level < 90:
        factors.append(
            {
                "title": "Very low submitted oxygen saturation",
                "description": (
                    "A low SpO₂ reading may be associated with weakness, fatigue, "
                    "confusion, or dizziness. These symptoms could make safe "
                    "movement more difficult."
                ),
                "direction": "higher",
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
                "direction": "higher",
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
                "direction": "lower",
            }
        )

    if movement_distance < 1:
        factors.append(
            {
                "title": "Very limited submitted movement distance",
                "description": (
                    "A low movement value may reflect reduced mobility, limited "
                    "activity, cautious movement, or difficulty completing normal "
                    "movement."
                ),
                "direction": "higher",
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
                "direction": "higher",
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
                "direction": "lower",
            }
        )

    if hrv < 20:
        factors.append(
            {
                "title": "Low submitted HRV value",
                "description": (
                    "The HRV value falls within the application's lower explanatory "
                    "range. HRV must be interpreted using the specific wearable, "
                    "measurement method, age, and personal baseline."
                ),
                "direction": "higher",
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
                "direction": "mixed",
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
                "direction": "lower",
            }
        )

    if glucose_level < 70:
        factors.append(
            {
                "title": "Low submitted blood glucose",
                "description": (
                    "Low blood glucose can be associated with shakiness, weakness, "
                    "confusion, or dizziness, which may affect safe movement."
                ),
                "direction": "higher",
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
                "direction": "higher",
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
                "direction": "lower",
            }
        )

    return factors


def get_result_reasoning(risk_level, factors):
    higher_risk_factors = [
        factor for factor in factors if factor["direction"] == "higher"
    ]

    lower_risk_factors = [
        factor for factor in factors if factor["direction"] == "lower"
    ]

    mixed_factors = [
        factor for factor in factors if factor["direction"] == "mixed"
    ]

    if risk_level == "high":
        explanation = (
            "The high-risk category may reflect the combined pattern of the "
            "submitted measurements, especially the values flagged below."
        )
        selected_factors = higher_risk_factors or mixed_factors or factors

    elif risk_level == "moderate":
        explanation = (
            "The possible-risk category may reflect a mixed pattern. Some "
            "measurements may indicate concern while others are not strongly "
            "flagged."
        )
        selected_factors = higher_risk_factors + mixed_factors

        if not selected_factors:
            selected_factors = factors

    else:
        explanation = (
            "The low-risk category may reflect the absence of multiple strongly "
            "flagged measurements in this submission."
        )
        selected_factors = lower_risk_factors or factors

    return explanation, selected_factors


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


def get_performance_dataframe(bundle):
    """
    Reads performance metrics if they were stored in the model bundle as either:
    bundle["performance"] or bundle["metrics"].
    """
    if not bundle:
        return None

    performance = bundle.get("performance") or bundle.get("metrics")

    if not isinstance(performance, dict) or not performance:
        return None

    rows = []

    for metric, value in performance.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_value = float(value)

            if numeric_value <= 1:
                numeric_value *= 100

            rows.append(
                {
                    "Metric": str(metric),
                    "Score": numeric_value,
                }
            )

    if not rows:
        return None

    return pd.DataFrame(rows)


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
            machine-learning risk estimate, model confidence context, and an
            explanation of the submitted values.
    </p>

    <p class="hero-note">
            Machine learning • Wearable sensors • Mobility research
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# APPLICATION TABS
# =========================================================
assessment_tab, performance_tab, about_tab = st.tabs(
    [
        "🩺 Assessment",
        "📊 Model Performance",
        "ℹ️ About the Project",
    ]
)


# =========================================================
# ASSESSMENT TAB
# =========================================================
with assessment_tab:
    st.markdown(
        """
        <div class="section-label">Fall-risk assessment</div>
        <div class="section-title">Enter wearable sensor measurements</div>
        <div class="section-description">
            Demonstration values are already entered so you can press the
            prediction button immediately. These are convenience defaults,
            not clinical recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = get_default_values(model_features, scaler)

    with st.form("fall_risk_form"):
        left_column, right_column = st.columns(2)

        with left_column:
            movement_distance = st.number_input(
                "Movement Distance",
                min_value=0.0,
                value=float(defaults["Distance"]),
                step=0.1,
                format="%.2f",
                help=(
                    "The amount of movement detected during the measurement "
                    "period. Use the same unit and measurement period as the "
                    "dataset used to train the model."
                ),
            )

            st.markdown(
                """
                <div class="reference-box">
                    <strong>Reference:</strong> No universal range. The correct
                    value depends on the wearable, unit, and recording period.
                </div>
                """,
                unsafe_allow_html=True,
            )

            hrv = st.number_input(
                "Heart Rate Variability (HRV)",
                min_value=0.0,
                value=float(defaults["HRV"]),
                step=0.1,
                format="%.2f",
                help=(
                    "HRV measures differences in time between consecutive "
                    "heartbeats. It should be interpreted using the same device, "
                    "metric, and personal baseline."
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
                value=float(defaults["Sugar level"]),
                step=1.0,
                format="%.1f",
                help=(
                    "Blood glucose interpretation depends on meal timing, medical "
                    "history, and the person's individual medical guidance."
                ),
            )

            st.markdown(
                """
                <div class="reference-box">
                    <strong>General context:</strong> 80–130 mg/dL before a meal;
                    below 180 mg/dL about 1–2 hours after eating. Personal targets
                    vary.
                </div>
                """,
                unsafe_allow_html=True,
            )

            oxygen_level = st.number_input(
                "Oxygen Saturation (SpO₂ %)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults["SpO2"]),
                step=0.1,
                format="%.1f",
                help=(
                    "A commonly cited normal range is approximately 95%–100%, "
                    "though medical conditions, elevation, and device accuracy "
                    "may affect readings."
                ),
            )

            st.markdown(
                """
                <div class="reference-box">
                    <strong>General reference:</strong> Approximately 95%–100%.
                    Medical conditions and elevation may affect an appropriate
                    reading.
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "Predict Fall Risk",
            use_container_width=True,
        )

    if submitted:
        if model_path is None:
            st.error(
                "No saved machine-learning model was found in the application files."
            )
            st.stop()

        if model_loading_error:
            st.error(
                f"The model file `{model_path}` was found, but it could not be loaded."
            )

            st.write("Error type and message:")
            st.code(model_loading_error)

            with st.expander("View complete traceback"):
                st.code(model_loading_traceback or "No traceback was captured.")

            st.stop()

        try:
            input_data = build_input_dataframe(
                movement_distance=movement_distance,
                hrv=hrv,
                glucose_level=glucose_level,
                oxygen_level=oxygen_level,
                feature_names=model_features,
            )

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
                "The model was loaded, but the submitted columns do not match "
                "the features used during model training."
            )
            st.code(str(error))
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

        with st.expander(
            "Why is the model confidence at this level?",
            expanded=True,
        ):
            st.write(get_confidence_explanation(confidence))

            st.caption(
                "Model confidence is the highest class probability produced by "
                "the model. It does not represent medical certainty or the exact "
                "chance that a person will fall."
            )

        factors = explain_measurements(
            movement_distance=movement_distance,
            hrv=hrv,
            glucose_level=glucose_level,
            oxygen_level=oxygen_level,
        )

        reasoning_text, selected_factors = get_result_reasoning(
            risk_level=risk_level,
            factors=factors,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)


        with st.expander(
            f"Why did the model return {risk_information['label']}?",
            expanded=True,
        ):
            st.write(reasoning_text)

            for factor in selected_factors:
                st.markdown(
                    f"""
                    <div class="factor">
                        <div class="factor-title">{factor["title"]}</div>

                    <p class="factor-description">
                            {factor["description"]}
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "These are rule-based explanations of the submitted measurements. "
                "They are not exact feature-attribution values from the trained "
                "model. The prediction may depend on interactions between several "
                "measurements."
            )

        with st.expander("Suggested caution"):
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
# MODEL PERFORMANCE TAB
# =========================================================
with performance_tab:
    st.markdown(
        """
        <div class="section-label">Evaluation</div>
        <div class="section-title">Model performance</div>
        <div class="section-description">
            Browse the evaluation visuals one at a time. SisFall is the main
            result, while cStick is shown as a smaller baseline comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sisfall_confusion_path = OUTPUTS_DIR / "sisfall_confusion_matrices.png"
    sisfall_trials_path = OUTPUTS_DIR / "sisfall_fall_vs_adl.png"
    sisfall_features_path = OUTPUTS_DIR / "sisfall_features_boxplots.png"
    cstick_confusion_path = OUTPUTS_DIR / "cstick_confusion_matrices.png"

    graph_options = [
        "sisfall_confusion",
        "sisfall_trials",
        "sisfall_features",
        "cstick_confusion",
    ]

    graph_specs = {
        "sisfall_confusion": {
            "title": "SIS Fall Confusion Matrix",
            "path": sisfall_confusion_path,
            "description": (
                "This confusion matrix shows how well the SisFall classifier "
                "separates fall from non-fall trials, including the elderly "
                "subset check. It is the main performance result for the "
                "project, and it is useful because it shows both correct "
                "predictions and the kinds of mistakes the model makes. "
                "Interpret the diagonal cells as correct predictions and the "
                "off-diagonal cells as errors."
            ),
        },
        "sisfall_trials": {
            "title": "SIS Fall Raw Signal",
            "path": sisfall_trials_path,
            "description": (
                "This plot compares a fall trial with an ADL trial using the "
                "raw sensor signal. It is useful because it shows the sharp "
                "impact spike in a fall versus the steadier pattern of normal "
                "movement. Look for the sudden spike in the fall trace and the "
                "more regular oscillation in the ADL trace."
            ),
        },
        "sisfall_features": {
            "title": "SIS Fall Feature Boxplots",
            "path": sisfall_features_path,
            "description": (
                "These boxplots show the engineered SisFall features split by "
                "class. They are useful because they explain why the model can "
                "learn the fall-versus-ADL separation from summary statistics. "
                "Tighter separation between the two boxes suggests a more "
                "predictive feature."
            ),
        },
        "cstick_confusion": {
            "title": "cStick Confusion Matrix",
            "path": cstick_confusion_path,
            "description": (
                "This confusion matrix shows the cStick baseline model, which "
                "is included for comparison because it is much easier than "
                "SisFall. It is useful as a reference point, but the perfect "
                "or near-perfect result should be interpreted cautiously due "
                "to the dataset's unusually clean separation. The diagonal "
                "cells show correct predictions; the off-diagonal cells would "
                "show any mix-ups."
            ),
        },
    }

    left_column, right_column = st.columns([3.3, 1.2], gap="large")

    with right_column:
        st.markdown(
            """
            <div class="graph-nav-panel">
                <div class="graph-nav-title">Graph Navigator</div>
                <div class="graph-nav-caption">
                    Select a single visualization to display on the left.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_graph = st.radio(
            label="Choose a graph",
            options=graph_options,
            index=0,
            format_func=lambda key: graph_specs[key]["title"],
            key="performance_graph_selector",
            label_visibility="collapsed",
        )

    with left_column:
        selected_spec = graph_specs[selected_graph]
        if selected_spec["path"].exists():
            st.image(
                str(selected_spec["path"]),
                caption=selected_spec["title"],
                use_container_width=True,
            )
            st.markdown(
                f"""
                <div class="reference-box" style="margin-top: 1rem; margin-bottom: 0;">
                    <strong>Description:</strong> {selected_spec["description"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"{selected_spec['title']} image was not found in outputs/.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    performance_dataframe = get_performance_dataframe(model_bundle)

    if performance_dataframe is not None:
        with st.expander("Optional metric summary"):
            st.bar_chart(
                performance_dataframe.set_index("Metric")["Score"],
                use_container_width=True,
            )

            st.dataframe(
                performance_dataframe,
                hide_index=True,
                use_container_width=True,
            )

            st.caption("Scores are displayed as percentages.")

    with st.expander("How to interpret model performance"):
        st.write(
            "Accuracy shows the share of correct predictions overall. Precision "
            "describes how often a predicted class was correct. Recall describes "
            "how many actual cases of a class the model identified. F1 score "
            "balances precision and recall. For fall-risk research, class-level "
            "performance is generally more informative than accuracy alone."
        )


# =========================================================
# ABOUT TAB
# =========================================================
with about_tab:
    st.markdown(
        """
        <div class="section-label">Project context</div>
        <div class="section-title">Why we chose this topic</div>
        <div class="section-description">
            Falls can reduce independence and quality of life, especially among
            older adults. Our project explores whether wearable sensor signals
            could support earlier recognition of mobility concerns and help
            researchers better understand patterns surrounding fall events.
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_one, card_two, card_three = st.columns(3)

    with card_one:
        st.markdown(
            """
            <div class="info-card">
                <h3>Problem</h3>

            <p>
                    Fall risk may develop before a visible fall occurs, but subtle
                    changes can be difficult to recognize consistently.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_two:
        st.markdown(
            """
            <div class="info-card">
                <h3>Research question</h3>

            <p>
                Can movement and physiological measurements help distinguish
                normal activity from near-fall or fall-related patterns?
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_three:
        st.markdown(
            """
            <div class="info-card">
                <h3>Goal</h3>

            <p>
                Build an interpretable research prototype that presents a risk
                estimate while clearly communicating uncertainty and limits.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Model status"):
        if model_path is None:
            st.warning("No supported model file was detected.")

        elif model_loading_error:
            st.error(
                f"Model detected but could not be loaded: {model_path}"
            )

        else:
            st.success(
                f"Model loaded successfully: {model_path}"
            )


# =========================================================
# DISCLAIMER
# =========================================================
st.markdown(
    """
    <div class="disclaimer">
        <strong>Research disclaimer:</strong>
        This application is intended for educational and research purposes.
        It is not a medical device and does not provide a diagnosis, treatment
        recommendation, or emergency assessment. Model confidence does not
        represent medical certainty. The displayed explanations are general,
        rule-based interpretations of the submitted values and may not represent
        the exact thresholds or internal reasoning learned by the machine-learning
        model. Medical concerns should be discussed with a qualified healthcare
        professional.
    </div>
    """,
    unsafe_allow_html=True,
)
