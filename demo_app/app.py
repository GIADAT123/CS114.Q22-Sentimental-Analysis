import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from inference import (
    predict_single_text,
    bundle
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mental Health Classification",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main container */
.block-container {
    padding-top: 3rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

h1 {
    font-size: 1.8rem !important;
    margin-top: 0rem !important;
    margin-bottom: 0.3rem !important;
    line-height: 1.3 !important;
}

/* Subtitle */
h3 {
    font-size: 1.1rem !important;
    margin-bottom: 0.2rem !important;
}

/* Text area */
textarea {
    font-size: 16px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}

/* Metric cards */
.pred-box {
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    background-color: #f8f9fa;
    text-align: center;
}

/* Prediction text */
.pred-label {
    font-size: 26px;
    font-weight: bold;
}

/* Confidence text */
.pred-conf {
    font-size: 22px;
    font-weight: bold;
}

/* Small labels */
.small-label {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 6px;
}
    
</style>
""", unsafe_allow_html=True)

# =====================================================
# BAR CHART FUNCTION
# =====================================================

LABEL_COLORS = {
    "anxiety": "#f39c12",
    "depression": "#e74c3c",
    "normal": "#27ae60",
    "suicidal": "#8e44ad"
}

def plot_confidence_bar(probs, labels):

    colors = [
        LABEL_COLORS.get(
            l.lower(),
            "#999999"
        )
        for l in labels
    ]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))

    bars = ax.bar(
        labels,
        probs,
        color=colors,
        width=0.6
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel(
        "Confidence",
        fontsize=10
    )

    ax.set_title(
        "Confidence Distribution",
        fontsize=14
    )

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9)

    for bar, v in zip(bars, probs):

        ax.text(
            bar.get_x() + bar.get_width()/2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            fontsize=9
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.3
    )

    plt.tight_layout()

    return fig

# =====================================================
# HEADER
# =====================================================

st.title("🧠 Mental Health Text Classification")

st.markdown("""
### TF-IDF Stacking Ensemble for Mental Health Classification
""")

# =====================================================
# MAIN LAYOUT
# =====================================================

left_col, right_col = st.columns([1.05, 1])

# =====================================================
# LEFT COLUMN
# =====================================================

with left_col:

    st.markdown("#### Input Text")

    text = st.text_area(
        "",
        height=140,
        placeholder="Example: I feel hopeless and anxious every day..."
    )

    predict_btn = st.button("🚀 Predict")

# =====================================================
# RIGHT COLUMN
# =====================================================

with right_col:

    st.markdown("#### 📊 Model Confidence")

    confidence_placeholder = st.empty()

# =====================================================
# PREDICTION
# =====================================================

if predict_btn:

    if text.strip() == "":

        st.warning("Please enter text.")

    else:

        with st.spinner("Analyzing text..."):

            result = predict_single_text(text)

        predicted_label = result["predicted_label"]

        class_list = bundle["class_list"]

        probs = []
        labels = []

        for cls in class_list:

            prob_col = f"prob_{cls}"

            labels.append(cls.capitalize())

            probs.append(
                float(result[prob_col])
            )

        # =================================================
        # CONFIDENCE TABLE
        # =================================================

        df = pd.DataFrame({
            "Label": labels,
            "Confidence": probs
        })

        df = df.sort_values(
            by="Confidence",
            ascending=False
        )

        def color_row(row):

            color_map = {
                "Anxiety": "#f39c12",
                "Depression": "#e74c3c",
                "Normal": "#27ae60",
                "Suicidal": "#8e44ad"
            }

            color = color_map.get(
                row["Label"],
                "#333333"
            )

            return [
                f"color: {color}; font-weight: bold;",
                f"color: {color}; font-weight: bold;"
            ]

        styled_df = (
            df.style
            .format({
                "Confidence": "{:.4f}"
            })
            .apply(color_row, axis=1)
        )

        confidence_placeholder.dataframe(
            styled_df,
            use_container_width=True,
            height=210
        )

        # =================================================
        # LOWER SECTION
        # =================================================

        lower_left, lower_right = st.columns([1, 1])

        # =============================================
        # PREDICTED LABEL
        # =============================================

        with lower_left:

            st.markdown("""
            <div class="small-label">
                Predicted Label
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">
                    {predicted_label.capitalize()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            top_conf = max(probs)

            st.markdown("""
            <div class="small-label">
                Predicted Label Confidence
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-conf">
                    {top_conf:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =============================================
        # BAR CHART
        # =============================================

        with lower_right:

            fig = plot_confidence_bar(
                probs,
                labels
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "CS114 - Machine Learning Project | TF-IDF Stacking Ensemble"
)