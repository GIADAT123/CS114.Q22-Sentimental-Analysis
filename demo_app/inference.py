# Import các thư viện dùng trong demo inference
import re
import ftfy
import emoji
import unicodedata
import joblib
import numpy as np
import pandas as pd

from wordsegment import load, segment
from sklearn.pipeline import Pipeline

load()


def fix_encoding(text: str) -> str:
    text = ftfy.fix_text(str(text)).lower()

    text = re.sub(
        r'(\+?\d{1,3}[\s.-]?)?(\d{3,4}[\s.-]?){2,3}',
        '',
        text
    )

    text = re.sub(
        r'(?m)^\s*\d+\s*[\.\)\-]\s*',
        '',
        text
    )

    return text


def convert_emoji(text: str, language: str = "en") -> str:
    text = emoji.demojize(str(text), language=language)
    text = re.sub(r":([a-zA-Z0-9_+-]+):", r" emoji_\1 ", text)
    return text


def clean_urls(text: str) -> str:
    pattern = r"https?://\S+|www\.\S+"
    return re.sub(pattern, " <url> ", str(text))


def clean_mentions(text: str) -> str:
    return re.sub(r"@\w+", " <username> ", str(text))


def split_camel(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text))
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    return text


def is_noise_hashtag(body: str) -> bool:
    body = str(body)

    if re.fullmatch(r'[\d\W_]+', body):
        return True

    if len(body) <= 2:
        return True

    if (
        re.fullmatch(r'[a-zA-Z0-9]{3,6}', body)
        and re.search(r'\d', body)
        and re.search(r'[a-zA-Z]', body)
    ):
        return True

    return False


def clean_hashtags(text: str) -> str:
    text = str(text)
    hashtags = re.findall(r'#(\w+)', text)

    for body in hashtags:
        if is_noise_hashtag(body):
            text = re.sub(r'#' + re.escape(body) + r'\b', '', text)
            continue

        camel_split = split_camel(body)
        words = segment(camel_split.lower())
        body_clean = ' '.join(words)

        text = re.sub(r'#' + re.escape(body) + r'\b', body_clean, text)

    return text.strip()


def normalize_repeated_chars(text: str, max_repeat: int = 2) -> str:
    text = str(text)
    pattern = rf"([a-zA-Z])\1{{{max_repeat},}}"
    replacement = r"\1" * max_repeat
    text = re.sub(pattern, replacement, text)
    return re.sub(r'\.{2,}', '...', text)


def normalize_unicode(text: str) -> str:
    text = str(text)

    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"\u2026", "...", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    return text


def normalize_to_ascii(text: str) -> str:
    normalized = unicodedata.normalize('NFD', str(text))
    ascii_text = normalized.encode('ascii', 'ignore').decode('utf-8')
    return ascii_text


def normalize_whitespace(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_text(
    text: str,
    emoji_language: str = "en",
    keep_case: bool = True,
) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = fix_encoding(text)
    text = convert_emoji(text, emoji_language)
    text = clean_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text)
    text = normalize_repeated_chars(text)
    text = normalize_unicode(text)
    text = normalize_to_ascii(text)
    text = normalize_whitespace(text)

    if not keep_case:
        text = text.lower()

    return text

# Input của hàm nên là text đã preprocess

def extract_advanced_features_for_demo(texts):

    if isinstance(texts, str):
        texts = [texts]

    df_out = pd.DataFrame({
        "text": [str(t) for t in texts]
    })

    raw_text = df_out["text"].astype(str)

    df_out["cleaned_text"] = raw_text.str.lower().str.strip()

    df_out["cleaned_text"] = df_out["cleaned_text"].apply(
        lambda x: re.sub(r"(.)\1{4,}", r"\1\1", x)
    )

    df_out["char_len"] = raw_text.str.len()

    df_out["word_count"] = raw_text.apply(
        lambda x: len(x.split())
    )

    df_out["avg_word_len"] = df_out["char_len"] / (df_out["word_count"] + 1e-5)

    df_out["uppercase_ratio"] = raw_text.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
    )

    df_out["i_pronoun_ratio"] = df_out["cleaned_text"].apply(
        lambda x: len(re.findall(r"\b(i|me|my|myself|mine)\b", x)) / (len(x.split()) + 1e-5)
    )

    df_out["negation_ratio"] = df_out["cleaned_text"].apply(
        lambda x: len(re.findall(r"\b(no|not|never|nothing|none|cannot|dont|don't|cant|can't)\b", x)) / (len(x.split()) + 1e-5)
    )

    df_out["ellipsis_count"] = raw_text.apply(
        lambda x: len(re.findall(r"\.\.\.", x))
    )

    df_out["exclamation_count"] = raw_text.apply(
        lambda x: x.count("!")
    )

    df_out["question_count"] = raw_text.apply(
        lambda x: x.count("?")
    )

    df_out["suicide_keyword_count"] = df_out["cleaned_text"].apply(
        lambda x: len(
            re.findall(
                r"\b(suicide|suicidal|kill myself|kms|kys|end my life|want to die|wanna die|die|overdose|hang myself)\b",
                x
            )
        )
    )

    feature_cols = [
        "cleaned_text",
        "word_count",
        "char_len",
        "avg_word_len",
        "uppercase_ratio",
        "i_pronoun_ratio",
        "negation_ratio",
        "ellipsis_count",
        "exclamation_count",
        "question_count",
        "suicide_keyword_count"
    ]

    return df_out[feature_cols]

# predict_tfidf_stacking: hàm inference chính dùng cho demo.
# aligned_predict_proba: đảm bảo xác suất các model luôn theo đúng thứ tự class_list.
# get_feature_pipeline: dựng lại pipeline preprocessor -> selector từ fold_bundle.

def aligned_predict_proba(model, X, class_names):
    """
    Đảm bảo xác suất predict_proba đúng thứ tự class_names.
    """
    proba = model.predict_proba(X)
    aligned = np.zeros((X.shape[0], len(class_names)), dtype=np.float32)

    for src_idx, cls in enumerate(model.classes_):
        dst_idx = np.where(class_names == cls)[0][0]
        aligned[:, dst_idx] = proba[:, src_idx]

    return aligned


def get_feature_pipeline(fold_bundle):

    fp = fold_bundle["feature_pipeline"]

    if isinstance(fp, dict):
        return Pipeline([
            ("preprocessor", fp["preprocessor"]),
            ("selector", fp["selector"])
        ])

    if hasattr(fp, "transform"):
        return fp

    raise TypeError(f"feature_pipeline không hợp lệ: {type(fp)}")


def predict_tfidf_stacking(texts, bundle, clean_fn=None):

    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t) for t in texts]

    if clean_fn is not None:
        processed_texts = [clean_fn(t) for t in texts]
    else:
        processed_texts = texts

    input_features = extract_advanced_features_for_demo(processed_texts)

    fold_bundles = bundle["fold_bundles"]
    meta_model = bundle["meta_model"]
    class_list = np.array(bundle["class_list"])
    class_weights = np.array(bundle["class_decision_weights"], dtype=np.float32)

    fold_meta_features = []

    for fold_bundle in fold_bundles:
        feature_pipeline = get_feature_pipeline(fold_bundle)
        models = fold_bundle["models"]

        X_fe = feature_pipeline.transform(input_features)

        svm_proba = aligned_predict_proba(
            models["svm"],
            X_fe,
            class_list
        )

        cnb_proba = aligned_predict_proba(
            models["cnb"],
            X_fe,
            class_list
        )

        lgb_proba = aligned_predict_proba(
            models["lgb"],
            X_fe,
            class_list
        )

        meta_features = np.hstack([
            svm_proba,
            cnb_proba,
            lgb_proba
        ])

        fold_meta_features.append(meta_features)

    meta_X = np.mean(fold_meta_features, axis=0)

    final_proba_raw = meta_model.predict_proba(meta_X)

    final_proba = np.zeros((len(texts), len(class_list)), dtype=np.float32)

    for src_idx, cls in enumerate(meta_model.classes_):
        dst_idx = np.where(class_list == cls)[0][0]
        final_proba[:, dst_idx] = final_proba_raw[:, src_idx]

    adjusted_scores = final_proba * class_weights

    pred_idx = np.argmax(adjusted_scores, axis=1)
    pred_labels = class_list[pred_idx]

    result_df = pd.DataFrame({
        "raw_text": texts,
        "processed_text": processed_texts,
        "predicted_label": pred_labels
    })

    for i, cls in enumerate(class_list):
        result_df[f"prob_{cls}"] = final_proba[:, i]
        result_df[f"adjusted_{cls}"] = adjusted_scores[:, i]

    return result_df

# Load bundle global:
MODEL_PATH = "models/tfidf_stacking_demo_bundle.joblib"
bundle = joblib.load(MODEL_PATH)

# Tạo wrapper đơn giản cho streamlit:
def predict_single_text(text):

    result = predict_tfidf_stacking(
        text,
        bundle,
        clean_fn=clean_text
    )

    return result.iloc[0]

