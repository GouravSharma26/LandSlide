"""KPI card component with glassmorphism dark navy aesthetics and custom badges."""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_kpi_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    highlight_color: str = "cyan",
    icon: Optional[str] = None,
) -> None:
    """Renders a single KPI card inside a Streamlit column."""
    color_class = f"kpi-highlight-{highlight_color}"
    icon_html = f"<span style='margin-right: 6px;'>{icon}</span>" if icon else ""

    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon_html}{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(cards: List[Dict[str, Any]]) -> None:
    """Renders a horizontal row of KPI cards across equal-width columns."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_kpi_card(
                label=card.get("label", ""),
                value=str(card.get("value", "")),
                subtext=card.get("subtext"),
                highlight_color=card.get("highlight_color", "cyan"),
                icon=card.get("icon"),
            )

