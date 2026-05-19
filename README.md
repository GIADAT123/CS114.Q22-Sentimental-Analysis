## Streamlit Demo App

This project includes a Streamlit-based demo application for Mental Health Text Classification using the TF-IDF Stacking Ensemble model.

### Features

- Input custom text for prediction
- Predict mental health categories:
  - Anxiety
  - Depression
  - Normal
  - Suicidal
- Display prediction confidence for all classes
- Visualize confidence distribution with a bar chart

---

## How to Run the Demo

### 1. Clone the repository

```bash
git clone https://github.com/GIADAT123/CS114.Q22-Sentimental-Analysis.git
cd CS114.Q22-Sentimental-Analysis
```

### 2. Install dependencies

```bash
pip install -r demo_app/requirements.txt
```

### 3. Run the Streamlit app

```bash
cd demo_app
streamlit run app.py
```

### 4. Open in browser

After running the command, Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser to use the demo application.

---

## Demo Model

The demo uses the pre-trained TF-IDF Stacking Ensemble model located at:

```text
demo_app/tfidf_stacking_demo_bundle.joblib
```

The model includes:
- TF-IDF preprocessing
- Multiple base classifiers
- Stacking meta-model
- Probability prediction for all labels
