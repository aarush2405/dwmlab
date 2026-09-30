"""Reusable Streamlit presentation elements."""

from __future__ import annotations

from html import escape

import streamlit as st

from config import ACCENT_CYAN, BG, CARD_BG, TEXT


def inject_css() -> None:
    st.markdown(f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    .block-container {{ padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1600px; }}
    section[data-testid="stSidebar"] {{ background: #0E1522; border-right: 1px solid rgba(255,255,255,.08); }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.25rem; }}
    h1, h2, h3, p, label {{ letter-spacing: -.015em; }}
    h1 {{ font-weight: 760; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: .75rem; margin: .7rem 0 1.2rem; }}
    div[data-testid="stMetric"], .glass-card {{
        background: linear-gradient(140deg, rgba(255,255,255,.04), rgba(255,255,255,.012)), {CARD_BG};
        border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
        padding: 1rem 1.1rem;
    }}
    .kpi-label {{ color: #B4BECD; font-size: .9rem; line-height: 1.3; }}
    .kpi-value {{ color: {TEXT}; font-size: clamp(1.45rem, 2vw, 1.9rem); font-weight: 720; margin-top: .42rem; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .kpi-accent {{ width: 35px; height: 3px; border-radius: 3px; margin-top: .9rem; }}
    div[data-baseweb="tab-list"] {{ gap: .35rem; }}
    button[data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: .65rem .95rem; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid rgba(255,255,255,.08); border-radius: 12px; }}
    div.stButton > button {{ border-radius: 10px; border: 1px solid rgba(255,255,255,.12); }}
    @media (max-width: 600px) {{
        .block-container h1 {{ font-size: 1.9rem; line-height: 1.17; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, accent: str = ACCENT_CYAN) -> str:
    return (
        '<div class="glass-card">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-accent" style="background:{escape(accent)}"></div>'
        '</div>'
    )
