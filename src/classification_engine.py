"""Order tier feature relevance, decision trees, and prediction sandbox."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import ACCENT_AMBER, ACCENT_CYAN, ACCENT_GREEN, ACCENT_ROSE, CATEGORIES, SIZES
from src.regression_engine import CATEGORICAL, FEATURES, NUMERIC, feature_frame

TIERS = ["Budget", "Standard", "Premium"]
COLORS = [ACCENT_GREEN, ACCENT_CYAN, ACCENT_ROSE]


def order_tiers(frame: pd.DataFrame) -> pd.Series:
    """Budget below $20; Standard $20 through $50; Premium above $50."""
    return pd.Series(np.select(
        [frame["total_price"].to_numpy() < 20, frame["total_price"].to_numpy() <= 50],
        ["Budget", "Standard"], default="Premium",
    ), index=frame.index, name="order_tier")


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", categories=[SIZES, CATEGORIES]), CATEGORICAL),
    ])


def _encoded_features(transformer: ColumnTransformer) -> list[str]:
    return list(transformer.get_feature_names_out())


def attribute_relevance(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 30 or order_tiers(frame).nunique() < 2:
        raise ValueError("At least 30 lines and two order tiers are needed for feature relevance.")
    X = feature_frame(frame)
    y = order_tiers(frame)
    transformer = _preprocessor()
    encoded = transformer.fit_transform(X)
    names = _encoded_features(transformer)
    discrete = [name.startswith("categorical__") for name in names]
    mi = mutual_info_classif(encoded, y, discrete_features=discrete, random_state=42)
    forest = RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=-1, class_weight="balanced")
    forest.fit(encoded, y)
    mapping = {}
    for name in names:
        raw = name.split("__", 1)[1]
        feature = next((column for column in CATEGORICAL if raw.startswith(column + "_")), raw)
        mapping[name] = feature
    scores = pd.DataFrame({
        "Feature": [mapping[name] for name in names],
        "Mutual Information": mi,
        "Random Forest Importance": forest.feature_importances_,
    })
    return scores.groupby("Feature", as_index=False).sum().sort_values("Random Forest Importance", ascending=False)


def relevance_chart(scores: pd.DataFrame) -> go.Figure:
    ordered = scores.sort_values("Random Forest Importance")
    fig = go.Figure()
    fig.add_bar(y=ordered["Feature"], x=ordered["Mutual Information"], name="Mutual Information",
                orientation="h", marker_color=ACCENT_CYAN)
    fig.add_bar(y=ordered["Feature"], x=ordered["Random Forest Importance"],
                name="Random Forest Importance", orientation="h", marker_color=ACCENT_AMBER)
    fig.update_layout(title="Attribute relevance for order tier", barmode="group", xaxis_title="Score",
                      legend=dict(orientation="h", y=1.12))
    return _style(fig)


@dataclass
class ClassificationResult:
    model: Pipeline
    report: pd.DataFrame
    confusion: np.ndarray
    y_test: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    auc: dict[str, float]


def train_decision_tree(
    frame: pd.DataFrame, max_depth: int = 5, min_samples_split: int = 8,
    criterion: str = "gini",
) -> ClassificationResult:
    if criterion not in {"gini", "entropy"}:
        raise ValueError("criterion must be gini or entropy")
    if max_depth < 1 or min_samples_split < 2:
        raise ValueError("Invalid decision tree hyperparameters")
    if len(frame) < 30:
        raise ValueError("At least 30 sales lines are needed for classification evaluation.")
    X = feature_frame(frame)
    y = order_tiers(frame)
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("At least two tiers with two examples each are needed for classification.")
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", DecisionTreeClassifier(
            max_depth=max_depth, min_samples_split=min_samples_split,
            criterion=criterion, class_weight="balanced", random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    report_dict = classification_report(y_test, predictions, labels=classes, output_dict=True, zero_division=0)
    report = pd.DataFrame([
        {"Tier": tier, "Precision": report_dict[tier]["precision"],
         "Recall": report_dict[tier]["recall"], "F1": report_dict[tier]["f1-score"],
         "Support": int(report_dict[tier]["support"])}
        for tier in classes
    ])
    report.loc[len(report)] = {
        "Tier": "Weighted average",
        "Precision": report_dict["weighted avg"]["precision"],
        "Recall": report_dict["weighted avg"]["recall"],
        "F1": report_dict["weighted avg"]["f1-score"],
        "Support": int(report_dict["weighted avg"]["support"]),
    }
    confusion = confusion_matrix(y_test, predictions, labels=classes, normalize="true")
    auc = {}
    for index, tier in enumerate(classes):
        binary = (np.asarray(y_test) == tier).astype(int)
        if binary.min() != binary.max():
            auc[tier] = float(roc_auc_score(binary, probabilities[:, index]))
    return ClassificationResult(model, report, confusion, np.asarray(y_test), probabilities, classes, auc)


def confusion_chart(result: ClassificationResult) -> go.Figure:
    labels = result.classes.tolist()
    fig = go.Figure(go.Heatmap(
        z=result.confusion, x=labels, y=labels, zmin=0, zmax=1,
        colorscale="Teal", text=np.round(result.confusion, 2), texttemplate="%{text:.2f}",
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:.1%}<extra></extra>",
        colorbar=dict(title="Row share"),
    ))
    fig.update_layout(title="Normalized confusion matrix", xaxis_title="Predicted tier", yaxis_title="Actual tier")
    fig.update_yaxes(autorange="reversed")
    return _style(fig)


def roc_chart(result: ClassificationResult) -> go.Figure:
    fig = go.Figure()
    for index, tier in enumerate(result.classes):
        binary = (result.y_test == tier).astype(int)
        if binary.min() == binary.max():
            continue
        false_positive, true_positive, _ = roc_curve(binary, result.probabilities[:, index])
        color = COLORS[TIERS.index(tier)] if tier in TIERS else ACCENT_CYAN
        fig.add_scatter(x=false_positive, y=true_positive, mode="lines",
                        name=f"{tier} (AUC {result.auc[tier]:.3f})", line=dict(color=color, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                    line=dict(color="rgba(255,255,255,0.45)", dash="dash"))
    fig.update_layout(title="One-vs-rest ROC curves", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return _style(fig)


def classify_order(
    model: Pipeline, quantity: int, unit_price: float, order_hour: int,
    pizza_size: str, pizza_category: str,
) -> dict[str, object]:
    if quantity < 1 or unit_price <= 0:
        raise ValueError("Quantity and unit price must be positive.")
    item = pd.DataFrame([{
        "quantity": quantity, "unit_price": unit_price, "order_hour": order_hour,
        "pizza_size": pizza_size, "pizza_category": pizza_category,
    }])
    probabilities = model.predict_proba(item)[0]
    return {
        "tier": str(model.predict(item)[0]),
        "probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist())),
        "basket_total": quantity * unit_price,
    }


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F9FAFB"),
                      margin=dict(l=20, r=20, t=50, b=30), height=400)
    return fig
