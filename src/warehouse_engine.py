"""Synthetic sales source, Pandas star schema, and OLAP operations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES, PIZZA_NAMES, SIZES

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "pizza_sales.csv"
REQUIRED_COLUMNS = {
    "order_id", "order_date", "order_time", "pizza_category", "pizza_size",
    "pizza_name", "quantity", "unit_price", "total_price",
}


def generate_sales_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible 2025 line items with weekend and meal rushes."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    weekday_weights = np.where(days.dayofweek == 4, 1.45, 1.0)
    weekday_weights = np.where(days.dayofweek == 5, 1.65, weekday_weights)
    weekday_weights = np.where(days.dayofweek == 6, 1.18, weekday_weights)
    chosen_days = rng.choice(days.to_numpy(), size=n_rows, p=weekday_weights / weekday_weights.sum())

    hours = np.arange(10, 24)
    hour_weights = np.array([0.035, 0.055, 0.12, 0.13, 0.10, 0.045, 0.04,
                             0.055, 0.07, 0.105, 0.105, 0.085, 0.04, 0.015])
    chosen_hours = rng.choice(hours, size=n_rows, p=hour_weights / hour_weights.sum())
    minutes = rng.integers(0, 60, n_rows)
    seconds = rng.integers(0, 60, n_rows)
    categories = rng.choice(CATEGORIES, size=n_rows, p=[0.35, 0.24, 0.22, 0.19])
    sizes = rng.choice(SIZES, size=n_rows, p=[0.13, 0.32, 0.33, 0.17, 0.05])
    names = [rng.choice(PIZZA_NAMES[category]) for category in categories]
    quantities = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.47, 0.29, 0.14, 0.07, 0.03])
    base_prices = {"Classic": 10.75, "Veggie": 11.50, "Supreme": 14.25, "Chicken": 13.25}
    size_increments = {"S": 0.0, "M": 2.25, "L": 4.50, "XL": 7.25, "XXL": 9.25}
    unit_prices = np.array([
        base_prices[category] + size_increments[size] + rng.choice([0.0, 0.75, 1.50, 2.25])
        for category, size in zip(categories, sizes)
    ]).clip(9.75, 25.50).round(2)

    frame = pd.DataFrame({
        "order_date": pd.to_datetime(chosen_days).strftime("%Y-%m-%d"),
        "order_time": [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(chosen_hours, minutes, seconds)],
        "pizza_category": categories,
        "pizza_size": sizes,
        "pizza_name": names,
        "quantity": quantities,
        "unit_price": unit_prices,
    })
    frame = frame.sort_values(["order_date", "order_time"], kind="stable").reset_index(drop=True)
    # About one in five lines shares its order with the preceding line.
    new_order = rng.random(n_rows) >= 0.20
    new_order[0] = True
    frame.insert(0, "order_id", [f"ORD-{number:05d}" for number in np.cumsum(new_order)])
    frame["total_price"] = (frame["quantity"] * frame["unit_price"]).round(2)
    return frame


def _validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Sales CSV is missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["order_date"] = pd.to_datetime(result["order_date"], errors="raise").dt.normalize()
    result["order_time"] = pd.to_datetime(result["order_time"].astype(str), format="%H:%M:%S", errors="raise").dt.strftime("%H:%M:%S")
    for column in ("quantity", "unit_price", "total_price"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.empty or (result["quantity"] <= 0).any() or (result["unit_price"] <= 0).any():
        raise ValueError("Sales CSV must contain positive quantity and unit_price values.")
    if not np.allclose(result["total_price"], result["quantity"] * result["unit_price"], atol=0.011):
        raise ValueError("Each total_price must equal quantity multiplied by unit_price.")
    return result


@lru_cache(maxsize=1)
def _load_sales_cached() -> pd.DataFrame:
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        generate_sales_data().to_csv(DATA_FILE, index=False)
    return _validate_source(pd.read_csv(DATA_FILE, dtype={"order_id": str}))


def load_sales_data() -> pd.DataFrame:
    """Return an isolated copy so callers cannot mutate the cached source."""
    return _load_sales_cached().copy()


@lru_cache(maxsize=1)
def _build_star_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = _load_sales_cached().copy().reset_index(drop=True)
    sales["date_key"] = sales["order_date"].dt.strftime("%Y%m%d").astype(int)
    sales["time_key"] = sales["order_time"].str.replace(":", "", regex=False).astype(int)
    dim_pizza = sales[["pizza_name", "pizza_category", "pizza_size"]].drop_duplicates().sort_values(
        ["pizza_category", "pizza_name", "pizza_size"]
    ).reset_index(drop=True)
    dim_pizza.insert(0, "pizza_key", np.arange(1, len(dim_pizza) + 1))
    sales = sales.merge(dim_pizza, on=["pizza_name", "pizza_category", "pizza_size"], validate="many_to_one")

    dim_time = sales[["time_key", "order_time"]].drop_duplicates().sort_values("time_key").reset_index(drop=True)
    dim_time["hour"] = dim_time["order_time"].str[:2].astype(int)
    dim_time["peak_period"] = np.select(
        [dim_time["hour"].between(12, 14), dim_time["hour"].between(19, 22)],
        ["Lunch", "Dinner"], default="Off-Peak",
    )

    dim_date = sales[["date_key", "order_date"]].drop_duplicates().sort_values("date_key").reset_index(drop=True)
    dim_date = dim_date.rename(columns={"order_date": "date"})
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    dim_date["is_weekend"] = dim_date["date"].dt.dayofweek >= 5
    dim_date["month_name"] = dim_date["date"].dt.month_name()
    dim_date["quarter"] = "Q" + dim_date["date"].dt.quarter.astype(str)

    fact = sales[["order_id", "date_key", "time_key", "pizza_key", "quantity", "unit_price", "total_price"]].copy()
    fact.insert(0, "sale_id", np.arange(1, len(fact) + 1))
    rng = np.random.default_rng(73)
    fact["margin_rate"] = rng.uniform(0.65, 0.75, len(fact)).round(4)
    return fact, dim_pizza, dim_time, dim_date


def build_star_schema() -> dict[str, pd.DataFrame]:
    """Build dimension and fact tables with surrogate keys."""
    fact, pizza, time, date = _build_star_cached()
    return {name: table.copy() for name, table in zip(
        ("Fact_Sales", "Dim_Pizza", "Dim_Time", "Dim_Date"), (fact, pizza, time, date)
    )}


@lru_cache(maxsize=1)
def _joined_cached() -> pd.DataFrame:
    fact, pizza, time, date = _build_star_cached()
    return (fact.merge(pizza, on="pizza_key", validate="many_to_one")
            .merge(time, on="time_key", validate="many_to_one")
            .merge(date, on="date_key", validate="many_to_one"))


def joined_sales() -> pd.DataFrame:
    return _joined_cached().copy()


def slice_dice(
    frame: pd.DataFrame,
    start_date: object | None = None,
    end_date: object | None = None,
    categories: list[str] | None = None,
    sizes: list[str] | None = None,
    weekend_only: bool = False,
    peak_only: bool = False,
) -> pd.DataFrame:
    """Filter a joined fact table across date, pizza, and time dimensions."""
    result = frame
    if start_date is not None:
        result = result.loc[result["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        result = result.loc[result["date"] <= pd.Timestamp(end_date)]
    if categories is not None:
        result = result.loc[result["pizza_category"].isin(categories)]
    if sizes is not None:
        result = result.loc[result["pizza_size"].isin(sizes)]
    if weekend_only:
        result = result.loc[result["is_weekend"]]
    if peak_only:
        result = result.loc[result["peak_period"].isin(["Lunch", "Dinner"])]
    return result.copy()


def roll_up(frame: pd.DataFrame, grain: str, dimensions: list[str] | None = None) -> pd.DataFrame:
    """Aggregate sales at Day, Week, Month, or Quarter grain plus dimensions."""
    if grain not in {"Day", "Week", "Month", "Quarter"}:
        raise ValueError("grain must be Day, Week, Month, or Quarter")
    dimensions = dimensions or []
    allowed = {"pizza_category", "pizza_size", "pizza_name", "peak_period", "day_of_week"}
    if not set(dimensions).issubset(allowed):
        raise ValueError("Unsupported OLAP dimension")
    columns = ["period", *dimensions, "revenue", "pizzas_sold", "orders", "gross_profit", "margin_pct"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    dates = pd.to_datetime(data["date"])
    if grain == "Day":
        data["period"] = dates.dt.strftime("%Y-%m-%d")
    elif grain == "Week":
        data["period"] = dates.dt.to_period("W-SUN").apply(lambda item: item.start_time.strftime("%Y-%m-%d"))
    elif grain == "Month":
        data["period"] = dates.dt.strftime("%Y-%m")
    else:
        data["period"] = dates.dt.to_period("Q").astype(str)
    data["gross_profit"] = data["total_price"] * data["margin_rate"]
    grouped = data.groupby(["period", *dimensions], observed=True, dropna=False).agg(
        revenue=("total_price", "sum"), pizzas_sold=("quantity", "sum"),
        orders=("order_id", "nunique"), gross_profit=("gross_profit", "sum"),
    ).reset_index()
    grouped["margin_pct"] = grouped["gross_profit"] / grouped["revenue"] * 100
    return grouped[columns].sort_values(["period", *dimensions]).reset_index(drop=True)
