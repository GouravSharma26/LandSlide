"""Himalayan Landslide Intelligence Dashboard — Main Application Entry Point.

Run using:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dashboard.config import ASSETS_DIR
from dashboard.components.sidebar import render_sidebar
from dashboard.pages import (
    overview,
    geographic_explorer,
    data_quality,
    environmental_analysis,
    event_detection,
    susceptibility_prediction,
    explainability_uncertainty,
    methodology_governance,
)

# Set page configuration
st.set_page_config(
    page_title="Himalayan Landslide Intelligence",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    """Loads and injects custom dark navy CSS stylesheet."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    """Main application router and lifecycle manager."""
    inject_custom_css()

    # Render sidebar and retrieve user selections
    filters = render_sidebar()
    selected_page = filters["page"]

    # 8-Section Router Table
    page_routers = {
        "Executive Overview": overview.render,
        "Geographic Explorer": geographic_explorer.render,
        "Data Catalog & Quality": data_quality.render,
        "Environmental Analysis": environmental_analysis.render,
        "Event Detection": event_detection.render,
        "Susceptibility Prediction": susceptibility_prediction.render,
        "Explainability & Uncertainty": explainability_uncertainty.render,
        "Research Methodology & Governance": methodology_governance.render,
    }

    render_fn = page_routers.get(selected_page)

    if render_fn:
        try:
            render_fn(filters)
        except Exception as e:
            st.error(f"Error rendering {selected_page}: {str(e)}")
            st.exception(e)
    else:
        st.warning(f"Page '{selected_page}' not recognized.")


if __name__ == "__main__":
    main()
