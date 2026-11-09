"""Unit tests for geospatial mapping components and Plotly/PyDeck compatibility."""

import pandas as pd
import pytest
import plotly.graph_objects as go
import pydeck as pdk
from unittest.mock import patch

from dashboard.components.maps import (
    render_observation_map,
    render_susceptibility_grid_map,
    render_pydeck_susceptibility_map,
)


@pytest.fixture
def sample_observation_data():
    return pd.DataFrame({
        "latitude": [30.1, 31.5, 32.2, 29.8],
        "longitude": [78.2, 77.4, 76.9, 79.1],
        "label": [1, 0, 1, 0],
        "event_date": ["2018-07-15", "2019-08-20", "2020-09-01", "2017-06-12"],
        "Landslide Trigger": ["Monsoon Rain", "None", "Monsoon Rain", "None"],
        "Location Accuracy": ["5km", "Known", "1km", "Known"],
    })


@pytest.fixture
def sample_grid_data():
    return pd.DataFrame({
        "latitude": [30.0, 30.5, 31.0, 31.5],
        "longitude": [77.0, 77.5, 78.0, 78.5],
        "susceptibility": [0.15, 0.45, 0.85, 0.60],
        "susceptibility_rf": [0.10, 0.50, 0.80, 0.65],
        "susceptibility_xgb": [0.20, 0.40, 0.90, 0.55],
        "rf_tree_uncertainty": [0.05, 0.12, 0.08, 0.15],
        "model_disagreement": [0.10, 0.10, 0.10, 0.10],
    })


def test_render_observation_map_valid(sample_observation_data):
    """Test observation map rendering with valid positive and negative data."""
    fig = render_observation_map(sample_observation_data, show_positives=True, show_negatives=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_render_observation_map_filtering(sample_observation_data):
    """Test observation map filtering for only positive or only negative observations."""
    fig_pos = render_observation_map(sample_observation_data, show_positives=True, show_negatives=False)
    assert isinstance(fig_pos, go.Figure)

    fig_neg = render_observation_map(sample_observation_data, show_positives=False, show_negatives=True)
    assert isinstance(fig_neg, go.Figure)


def test_render_observation_map_empty():
    """Test observation map returns valid fallback figure on empty data."""
    fig = render_observation_map(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert "No Data" in fig.layout.title.text

    fig_missing = render_observation_map(pd.DataFrame({"x": [1, 2]}))
    assert isinstance(fig_missing, go.Figure)
    assert "No Data" in fig_missing.layout.title.text


def test_render_susceptibility_grid_map_valid(sample_grid_data):
    """Test density susceptibility map renders with valid grid data."""
    fig = render_susceptibility_grid_map(sample_grid_data, value_column="susceptibility")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_render_susceptibility_grid_map_attributes(sample_grid_data):
    """Test density map renders for different attributes without error."""
    for col in ["susceptibility_rf", "susceptibility_xgb", "rf_tree_uncertainty", "model_disagreement"]:
        fig = render_susceptibility_grid_map(sample_grid_data, value_column=col)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


def test_render_susceptibility_grid_map_empty():
    """Test susceptibility grid map gracefully handles empty DataFrame."""
    fig = render_susceptibility_grid_map(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_plotly_6_compatibility_simulation(sample_observation_data, sample_grid_data):
    """Simulate Plotly 6 environment where px.density_mapbox and px.scatter_mapbox are removed.
    
    Verifies that render_susceptibility_grid_map and render_observation_map execute
    cleanly using px.density_map and px.scatter_map without raising AttributeError.
    """
    import plotly.express as px

    # In Plotly 6, density_mapbox and scatter_mapbox are removed
    with patch.object(px, "density_mapbox", side_effect=AttributeError("module 'plotly.express' has no attribute 'density_mapbox'"), create=True):
        with patch.object(px, "scatter_mapbox", side_effect=AttributeError("module 'plotly.express' has no attribute 'scatter_mapbox'"), create=True):
            # Must not raise AttributeError:
            fig_obs = render_observation_map(sample_observation_data)
            assert isinstance(fig_obs, go.Figure)

            fig_grid = render_susceptibility_grid_map(sample_grid_data, value_column="susceptibility")
            assert isinstance(fig_grid, go.Figure)


def test_render_pydeck_susceptibility_map(sample_grid_data):
    """Test 3D PyDeck map rendering."""
    deck = render_pydeck_susceptibility_map(sample_grid_data, elevation_scale=40)
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 1

    deck_empty = render_pydeck_susceptibility_map(pd.DataFrame())
    assert isinstance(deck_empty, pdk.Deck)
