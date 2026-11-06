"""Geospatial mapping components using Plotly Mapbox and PyDeck."""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from dashboard.config import HIMALAYAN_CENTER, THEME_COLORS, ROI_BOUNDS


def render_observation_map(
    df: pd.DataFrame,
    show_positives: bool = True,
    show_negatives: bool = True,
    height: int = 550,
) -> go.Figure:
    """Renders interactive Plotly map showing positive landslide events and control negatives."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_COLORS["card_bg"],
            title="Geospatial Map (No Data)",
        )
        return fig

    # Filter by label selection
    labels_to_show = []
    if show_positives:
        labels_to_show.append(1)
    if show_negatives:
        labels_to_show.append(0)

    plot_df = df[df["label"].isin(labels_to_show)].copy() if "label" in df.columns else df.copy()

    if "label" in plot_df.columns:
        plot_df["Class"] = plot_df["label"].map({1: "Landslide Event (Positive)", 0: "Stable Control (Hard Negative)"})
        color_map = {
            "Landslide Event (Positive)": THEME_COLORS["danger"],
            "Stable Control (Hard Negative)": THEME_COLORS["primary"],
        }
    else:
        plot_df["Class"] = "Observation"
        color_map = {"Observation": THEME_COLORS["primary"]}

    # Build custom hover data
    hover_cols = ["latitude", "longitude"]
    if "event_date" in plot_df.columns:
        hover_cols.append("event_date")
    if "Landslide Trigger" in plot_df.columns:
        hover_cols.append("Landslide Trigger")
    if "Location Accuracy" in plot_df.columns:
        hover_cols.append("Location Accuracy")

    fig = px.scatter_mapbox(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="Class",
        color_discrete_map=color_map,
        hover_name="Class",
        hover_data=hover_cols,
        zoom=5,
        center={"lat": HIMALAYAN_CENTER["lat"], "lon": HIMALAYAN_CENTER["lon"]},
        mapbox_style="carto-darkmatter",
        height=height,
    )

    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME_COLORS["card_bg"],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=dict(
            text="Himalayan Landslide Distribution & Control Sites",
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            y=0.97,
        ),
        legend=dict(
            bgcolor="rgba(28, 37, 65, 0.85)",
            bordercolor=THEME_COLORS["card_border"],
            borderwidth=1,
            font=dict(color="#E2E8F0", size=11),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
    )
    return fig


def render_susceptibility_grid_map(
    grid_df: pd.DataFrame,
    value_column: str = "susceptibility",
    title: str = "Himalayan Regional Susceptibility Grid (0.5° Resolution)",
    color_scale: str = "YlOrRd",
    height: int = 580,
) -> go.Figure:
    """Renders regional 0.5-degree susceptibility grid heatmap over the Himalayas."""
    if grid_df.empty or "latitude" not in grid_df.columns or "longitude" not in grid_df.columns:
        fig = go.Figure()
        return fig

    hover_data = {
        "latitude": ":.2f",
        "longitude": ":.2f",
        value_column: ":.3f",
    }
    for col in ["susceptibility", "susceptibility_rf", "susceptibility_xgb", "rf_tree_uncertainty", "model_disagreement"]:
        if col in grid_df.columns and col != value_column:
            hover_data[col] = ":.3f"

    fig = px.density_mapbox(
        grid_df,
        lat="latitude",
        lon="longitude",
        z=value_column,
        radius=28,
        center={"lat": HIMALAYAN_CENTER["lat"], "lon": HIMALAYAN_CENTER["lon"]},
        zoom=4.8,
        mapbox_style="carto-darkmatter",
        color_continuous_scale=color_scale,
        range_color=[0.0, 1.0] if "susceptibility" in value_column else None,
        hover_data=hover_data,
        height=height,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME_COLORS["card_bg"],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=dict(
            text=title,
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            y=0.97,
        ),
        coloraxis_colorbar=dict(
            title=dict(text=value_column.replace("_", " ").title(), font=dict(color="#FFFFFF", size=12)),
            tickfont=dict(color="#A0AEC0", size=10),
            bgcolor="rgba(28, 37, 65, 0.8)",
            bordercolor=THEME_COLORS["card_border"],
            borderwidth=1,
            len=0.7,
        ),
    )
    return fig


def render_pydeck_susceptibility_map(
    grid_df: pd.DataFrame,
    value_column: str = "susceptibility",
    elevation_scale: float = 80000,
) -> pdk.Deck:
    """Renders 3D PyDeck column map visualizing susceptibility or uncertainty across the Himalayas."""
    if grid_df.empty:
        return pdk.Deck()

    plot_df = grid_df.copy()
    
    # Calculate RGB colors based on value (0 to 1)
    vals = plot_df[value_column].fillna(0).values
    v_norm = np.clip(vals, 0.0, 1.0)
    
    # Red-Orange-Yellow or Purple-Cyan gradient
    r = (v_norm * 240 + 15).astype(int)
    g = ((1 - v_norm) * 180 + 30).astype(int)
    b = ((1 - v_norm) * 220).astype(int)
    
    plot_df["color_r"] = r
    plot_df["color_g"] = g
    plot_df["color_b"] = b
    plot_df["elevation"] = v_norm * elevation_scale

    layer = pdk.Layer(
        "ColumnLayer",
        data=plot_df,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=25000,  # 25 km column radius for 0.5-deg cells
        get_fill_color=["color_r", "color_g", "color_b", 200],
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=HIMALAYAN_CENTER["lat"],
        longitude=HIMALAYAN_CENTER["lon"],
        zoom=4.7,
        pitch=45,
        bearing=15,
    )

    tooltip = {
        "html": f"<b>Coordinates:</b> {{latitude}}°N, {{longitude}}°E<br>"
                f"<b>{value_column}:</b> {{{value_column}}}<br>"
                "<b>Resolution:</b> 0.5° (~55 km)",
        "style": {
            "backgroundColor": "#0D1B2A",
            "color": "#FFFFFF",
            "fontSize": "12px",
            "padding": "10px",
            "border": "1px solid #48E5C2",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10",
    )

