"""Price regression models, diagnostics, and basket inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ACCENT_CYAN, ACCENT_GREEN, ACCENT_ROSE, CATEGORIES, SIZES

FEATURES = ["quantity", "unit_price", "order_hour", "pizza_size", "pizza_category"]
NUMERIC = ["quantity", "unit_price", "order_hour"]
CATEGORICAL = ["pizza_size", "pizza_category"]


@dataclass
class RegressionResult:
    metrics: pd.DataFrame
    models: dict[str, Pipeline]
    predictions: dict[str, np.ndarray]
    y_test: np.ndarray
    test_rows: pd.DataFrame


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["quantity", "unit_price", "pizza_size", "pizza_category"]].copy()
    result["order_hour"] = frame["hour"].to_numpy()
    return result[FEATURES]


def _pipeline(estimator: object) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", categories=[SIZES, CATEGORIES]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", transformer), ("model", estimator)])


def train_regressors(frame: pd.DataFrame) -> RegressionResult:
    if len(frame) < 30:
        raise ValueError("At least 30 sales lines are needed for regression evaluation.")
    X = feature_frame(frame)
    y = frame["total_price"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name, estimator in candidates.items():
        model = _pipeline(estimator)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        models[name] = model
        predictions[name] = predicted
        r2 = r2_score(y_test, predicted)
        p = model.named_steps["preprocess"].transform(X_test).shape[1]
        n = len(y_test)
        adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        rows.append({
            "Model": name, "R2": r2, "Adjusted R2": adjusted,
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
            "MAPE (%)": mean_absolute_percentage_error(y_test, predicted) * 100,
        })
    return RegressionResult(pd.DataFrame(rows), models, predictions, y_test, X_test)


def residual_plot(result: RegressionResult, model_name: str) -> go.Figure:
    fitted = result.predictions[model_name]
    residuals = result.y_test - fitted
    fig = go.Figure(go.Scatter(
        x=fitted, y=residuals, mode="markers", marker=dict(color=ACCENT_CYAN, opacity=0.65, size=7),
        customdata=result.y_test, hovertemplate="Fitted $%{x:.2f}<br>Residual $%{y:.2f}<br>Actual $%{customdata:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=ACCENT_ROSE, line_dash="dash")
    fig.update_layout(title=f"Residuals · {model_name}", xaxis_title="Fitted price ($)", yaxis_title="Residual ($)")
    return _style(fig)


def qq_plot(result: RegressionResult, model_name: str) -> go.Figure:
    residuals = result.y_test - result.predictions[model_name]
    theoretical, ordered = stats.probplot(residuals, dist="norm", fit=False)
    slope, intercept, _ = stats.probplot(residuals, dist="norm", fit=True)[1]
    fig = go.Figure()
    fig.add_scatter(x=theoretical, y=ordered, mode="markers", name="Residual quantiles",
                    marker=dict(color=ACCENT_GREEN, opacity=0.7))
    endpoints = np.array([min(theoretical), max(theoretical)])
    fig.add_scatter(x=endpoints, y=slope * endpoints + intercept, mode="lines", name="Normal reference",
                    line=dict(color=ACCENT_ROSE, dash="dash"))
    fig.update_layout(title=f"Residual QQ · {model_name}", xaxis_title="Theoretical normal quantiles",
                      yaxis_title="Observed residual quantiles")
    return _style(fig)


def predict_basket_price(
    model: Pipeline, quantity: int, base_price: float, size_uplift: float,
    surge_multiplier: float, pizza_size: str, pizza_category: str, order_hour: int,
) -> dict[str, float]:
    """Infer a basket total after applying the chosen uplift and surge to unit price."""
    if quantity < 1 or base_price <= 0 or size_uplift < 0 or surge_multiplier <= 0:
        raise ValueError("Basket inputs must be positive; size uplift may be zero.")
    effective_unit_price = (base_price + size_uplift) * surge_multiplier
    item = pd.DataFrame([{
        "quantity": quantity, "unit_price": effective_unit_price, "order_hour": order_hour,
        "pizza_size": pizza_size, "pizza_category": pizza_category,
    }])
    return {
        "effective_unit_price": effective_unit_price,
        "model_prediction": max(0.0, float(model.predict(item)[0])),
        "arithmetic_total": quantity * effective_unit_price,
    }


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F9FAFB"),
                      margin=dict(l=20, r=20, t=50, b=30), height=380)
    return fig
