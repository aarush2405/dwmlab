"""Pizza Sales Modern Data Warehouse & Predictive Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_GREEN, ACCENT_ROSE,
    CATEGORIES, SIZES,
)
from src.analytics import category_waterfall, executive_kpis, hourly_sales_heatmap, pareto_chart
from src.classification_engine import (
    attribute_relevance, classify_order, confusion_chart, order_tiers,
    relevance_chart, roc_chart, train_decision_tree,
)
from src.components import inject_css, metric_card
from src.regression_engine import predict_basket_price, qq_plot, residual_plot, train_regressors
from src.warehouse_engine import build_star_schema, joined_sales, roll_up, slice_dice

st.set_page_config(
    page_title="Pizza MDW & Predictive Intelligence", layout="wide",
    initial_sidebar_state="expanded", page_icon="🍕",
)
inject_css()


@st.cache_data(show_spinner="Training regression models…")
def cached_regression(frame: pd.DataFrame):
    return train_regressors(frame)


@st.cache_data(show_spinner="Training decision tree…")
def cached_tree(frame: pd.DataFrame, depth: int, split: int, criterion: str):
    return train_decision_tree(frame, depth, split, criterion)


@st.cache_data(show_spinner="Ranking features…")
def cached_relevance(frame: pd.DataFrame):
    return attribute_relevance(frame)


def set_preset(name: str, first_date, last_date) -> None:
    st.session_state.date_range = (first_date, last_date)
    st.session_state.categories = CATEGORIES.copy()
    st.session_state.sizes = SIZES.copy()
    st.session_state.preset = name


def render_sidebar(source: pd.DataFrame) -> pd.DataFrame:
    first_date = source["date"].min().date()
    last_date = source["date"].max().date()
    if "date_range" not in st.session_state:
        st.session_state.date_range = (first_date, last_date)
    if "categories" not in st.session_state:
        st.session_state.categories = CATEGORIES.copy()
    if "sizes" not in st.session_state:
        st.session_state.sizes = SIZES.copy()
    if "preset" not in st.session_state:
        st.session_state.preset = "All Time"

    st.sidebar.title("🍕 Pizza MDW")
    st.sidebar.caption("2025 sales intelligence workspace")
    st.sidebar.subheader("Slice the warehouse")
    date_range = st.sidebar.date_input(
        "Order date", min_value=first_date, max_value=last_date,
        key="date_range", format="YYYY-MM-DD",
    )
    categories = st.sidebar.multiselect("Category", CATEGORIES, key="categories")
    sizes = st.sidebar.multiselect("Size", SIZES, key="sizes")
    st.sidebar.caption("Quick presets")
    st.sidebar.button("All Time", use_container_width=True, on_click=set_preset,
                      args=("All Time", first_date, last_date))
    st.sidebar.button("Weekend Rush", use_container_width=True, on_click=set_preset,
                      args=("Weekend Rush", first_date, last_date))
    st.sidebar.button("Peak Hours", use_container_width=True, on_click=set_preset,
                      args=("Peak Hours", first_date, last_date))
    preset = st.session_state.preset
    st.sidebar.info(f"Preset: {preset}")
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.sidebar.warning("Select both a start and end date.")
        return source.iloc[0:0].copy()
    return slice_dice(
        source, date_range[0], date_range[1], categories, sizes,
        weekend_only=preset == "Weekend Rush",
        peak_only=preset in {"Weekend Rush", "Peak Hours"},
    )


def render_overview(frame: pd.DataFrame) -> None:
    st.subheader("Executive Overview")
    if frame.empty:
        st.info("No sales match the current filters. Broaden the date or pizza selection.")
        return
    kpis = executive_kpis(frame)
    cards = [
        ("Gross Revenue", f"${kpis['Gross Revenue']:,.2f}", ACCENT_GREEN),
        ("Pizzas Sold", f"{kpis['Total Pizzas Sold']:,}", ACCENT_CYAN),
        ("Unique Orders", f"{kpis['Total Unique Orders']:,}", ACCENT_AMBER),
        ("Average Order Value", f"${kpis['Average Order Value']:,.2f}", ACCENT_ROSE),
        ("Average Margin", f"{kpis['Average Margin %']:.1f}%", ACCENT_GREEN),
    ]
    st.markdown('<div class="kpi-grid">' + ''.join(metric_card(*card) for card in cards) + '</div>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(category_waterfall(frame), use_container_width=True)
    with right:
        st.plotly_chart(hourly_sales_heatmap(frame), use_container_width=True)
    st.caption("Margin is an illustrative 65–75% assumption assigned per sales line.")


def render_olap(frame: pd.DataFrame, schema: dict[str, pd.DataFrame]) -> None:
    st.subheader("OLAP & Data Warehouse")
    facts = len(schema["Fact_Sales"])
    st.caption(
        f"Star schema · {facts:,} fact rows · {len(schema['Dim_Pizza'])} pizzas · "
        f"{len(schema['Dim_Time'])} times · {len(schema['Dim_Date'])} dates"
    )
    with st.expander("Inspect star schema tables"):
        selected_table = st.selectbox("Table", list(schema), key="schema_table")
        st.dataframe(schema[selected_table].head(100), use_container_width=True, hide_index=True)
    if frame.empty:
        st.info("No sales match the current filters.")
        return
    col_grain, col_dimensions = st.columns([1, 2])
    with col_grain:
        grain = st.selectbox("Time hierarchy", ["Day", "Week", "Month", "Quarter"])
    with col_dimensions:
        dimensions = st.multiselect(
            "Slice/dice dimensions",
            ["pizza_category", "pizza_size", "pizza_name", "peak_period", "day_of_week"],
            default=["pizza_category"],
        )
    cube = roll_up(frame, grain, dimensions)
    st.dataframe(cube, use_container_width=True, hide_index=True,
                 column_config={
                     "revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                     "gross_profit": st.column_config.NumberColumn("Gross profit", format="$%.2f"),
                     "margin_pct": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
                 })
    timeline = cube.groupby("period", as_index=False)["revenue"].sum()
    fig = go.Figure(go.Scatter(x=timeline["period"], y=timeline["revenue"], mode="lines+markers",
                               line=dict(color=ACCENT_CYAN, width=3), fill="tozeroy",
                               fillcolor="rgba(6,182,212,.12)"))
    fig.update_layout(title=f"Revenue drill-down · {grain}", xaxis_title=grain, yaxis_title="Revenue ($)",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=330)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(pareto_chart(frame), use_container_width=True)
    st.caption("A line item is the fact grain. Distinct orders in grouped rows may overlap across dimension groups.")


def training_frame(filtered: pd.DataFrame, source: pd.DataFrame, minimum: int, needs_tiers: bool = False) -> pd.DataFrame:
    sufficient = len(filtered) >= minimum
    if needs_tiers and sufficient:
        counts = order_tiers(filtered).value_counts()
        sufficient = len(counts) >= 2 and counts.min() >= 2
    if sufficient:
        st.caption(f"Models trained on {len(filtered):,} filtered sales lines; 25% held out for evaluation.")
        return filtered
    st.warning("Current filters leave too little training data for this lab. Models use the full warehouse dataset.")
    return source


def render_regression(filtered: pd.DataFrame, source: pd.DataFrame) -> None:
    st.subheader("Price Regression Lab")
    frame = training_frame(filtered, source, 60)
    result = cached_regression(frame)
    st.dataframe(result.metrics, use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.3f")
                                for name in ["R2", "Adjusted R2", "MAE", "RMSE", "MAPE (%)"]})
    selected = st.selectbox("Inspect model diagnostics", list(result.models), index=2)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(residual_plot(result, selected), use_container_width=True)
    with right:
        st.plotly_chart(qq_plot(result, selected), use_container_width=True)

    st.markdown("#### Interactive Basket Price Calculator")
    with st.form("basket_calculator"):
        c1, c2, c3, c4 = st.columns(4)
        quantity = c1.number_input("Quantity", min_value=1, max_value=20, value=2, step=1)
        base_price = c2.number_input("Base unit price ($)", min_value=0.01, max_value=100.0, value=12.0, step=0.25)
        uplift = c3.number_input("Size uplift ($)", min_value=0.0, max_value=50.0, value=2.0, step=0.25)
        surge = c4.number_input("Surge multiplier", min_value=0.1, max_value=3.0, value=1.0, step=0.05)
        c5, c6, c7 = st.columns(3)
        size = c5.selectbox("Size", SIZES, index=1, key="reg_size")
        category = c6.selectbox("Category", CATEGORIES, key="reg_category")
        hour = c7.slider("Order hour", 10, 23, 19, key="reg_hour")
        submitted = st.form_submit_button("Calculate basket price", use_container_width=True)
    if submitted:
        prediction = predict_basket_price(result.models[selected], quantity, base_price, uplift,
                                          surge, size, category, hour)
        x, y, z = st.columns(3)
        x.metric("Effective unit price", f"${prediction['effective_unit_price']:,.2f}")
        y.metric("Model prediction", f"${prediction['model_prediction']:,.2f}")
        z.metric("Arithmetic basket total", f"${prediction['arithmetic_total']:,.2f}")
        st.caption("Surge changes the input unit price. Predictions outside the training price range are extrapolations.")


def render_classification(filtered: pd.DataFrame, source: pd.DataFrame) -> None:
    st.subheader("Tree Classification & Mining")
    frame = training_frame(filtered, source, 60, needs_tiers=True)
    st.caption("Tiers: Budget < \\$20 · Standard \\$20–\\$50 · Premium > \\$50")
    score = cached_relevance(frame)
    st.plotly_chart(relevance_chart(score), use_container_width=True)
    st.dataframe(score, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    depth = c1.slider("Maximum tree depth", 2, 15, 5)
    split = c2.slider("Minimum samples to split", 2, 30, 8)
    criterion = c3.selectbox("Split criterion", ["gini", "entropy"])
    result = cached_tree(frame, depth, split, criterion)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_chart(result), use_container_width=True)
    with right:
        st.plotly_chart(roc_chart(result), use_container_width=True)
    st.markdown("#### Classification report")
    st.dataframe(result.report, use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.3f")
                                for name in ["Precision", "Recall", "F1"]})

    st.markdown("#### Live Order Tier Classifier")
    with st.form("tier_classifier"):
        a, b, c = st.columns(3)
        quantity = a.number_input("Quantity", 1, 20, 2, key="tier_quantity")
        unit_price = b.number_input("Unit price ($)", 0.01, 100.0, 15.0, step=0.25, key="tier_price")
        hour = c.slider("Order hour", 10, 23, 19, key="tier_hour")
        d, e = st.columns(2)
        size = d.selectbox("Size", SIZES, index=1, key="tier_size")
        category = e.selectbox("Category", CATEGORIES, key="tier_category")
        submitted = st.form_submit_button("Classify order", use_container_width=True)
    if submitted:
        prediction = classify_order(result.model, quantity, unit_price, hour, size, category)
        st.success(f"Predicted tier: **{prediction['tier']}** · Basket total: ${prediction['basket_total']:,.2f}")
        probability_frame = pd.DataFrame({
            "Tier": list(prediction["probabilities"]),
            "Probability": [value * 100 for value in prediction["probabilities"].values()],
        })
        st.dataframe(probability_frame, use_container_width=True, hide_index=True,
                     column_config={"Probability": st.column_config.ProgressColumn(
                         "Class probability", min_value=0, max_value=100, format="%.1f%%"
                     )})
        st.caption("Tree probabilities reflect the trained model; the tier definition uses the stated dollar thresholds.")


def main() -> None:
    source = joined_sales()
    schema = build_star_schema()
    filtered = render_sidebar(source)
    st.title("Pizza MDW & Predictive Intelligence")
    st.caption("Sales decisions, multidimensional analysis, and applied machine learning in one workspace")
    tabs = st.tabs([
        "Executive Overview", "OLAP & Data Warehouse", "Price Regression Lab", "Tree Classification & Mining",
    ])
    with tabs[0]:
        render_overview(filtered)
    with tabs[1]:
        render_olap(filtered, schema)
    with tabs[2]:
        render_regression(filtered, source)
    with tabs[3]:
        render_classification(filtered, source)


if __name__ == "__main__":
    main()
