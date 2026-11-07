"""Visualization themes and layout builders for Plotly and PyDeck figures."""

from typing import Dict, Any, Optional
import plotly.graph_objects as go
from dashboard.config import THEME_COLORS


def get_base_layout(
    title: str = "",
    height: int = 420,
    show_legend: bool = True,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates standard layout dictionary for dark navy dashboard plots."""
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": THEME_COLORS["card_bg"],
        "plot_bgcolor": THEME_COLORS["card_bg"],
        "font": {
            "family": "Inter, -apple-system, sans-serif",
            "color": THEME_COLORS["text_primary"],
            "size": 12,
        },
        "title": {
            "text": title,
            "font": {
                "size": 15,
                "color": THEME_COLORS["text_primary"],
                "family": "Inter, sans-serif",
            },
            "x": 0.02,
            "y": 0.95,
        },
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
        "height": height,
        "showlegend": show_legend,
        "legend": {
            "bgcolor": "rgba(28, 37, 65, 0.7)",
            "bordercolor": THEME_COLORS["card_border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": THEME_COLORS["text_secondary"]},
        },
        "xaxis": {
            "gridcolor": THEME_COLORS["grid_color"],
            "linecolor": THEME_COLORS["card_border"],
            "tickcolor": THEME_COLORS["card_border"],
            "tickfont": {"size": 11, "color": THEME_COLORS["text_secondary"]},
            "title": {"text": x_title, "font": {"size": 12, "color": THEME_COLORS["text_secondary"]}} if x_title else None,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": THEME_COLORS["grid_color"],
            "linecolor": THEME_COLORS["card_border"],
            "tickcolor": THEME_COLORS["card_border"],
            "tickfont": {"size": 11, "color": THEME_COLORS["text_secondary"]},
            "title": {"text": y_title, "font": {"size": 12, "color": THEME_COLORS["text_secondary"]}} if y_title else None,
            "zeroline": False,
        },
        "hoverlabel": {
            "bgcolor": "#0D1B2A",
            "bordercolor": THEME_COLORS["primary"],
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#FFFFFF"},
        },
    }
    return layout


def apply_custom_theme(fig: go.Figure, **kwargs) -> go.Figure:
    """Applies the custom dark navy styling directly to a Plotly figure."""
    base_layout = get_base_layout(**kwargs)
    fig.update_layout(**base_layout)
    return fig

