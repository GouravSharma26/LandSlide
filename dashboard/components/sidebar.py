"""Sidebar navigation and filter panel."""

import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """Renders the simple, clean sidebar navigation and filter controls."""
    with st.sidebar:
        st.markdown("### Early Landslide Risk Detection")
        st.caption("Visual Terrain Changes & Remote Sensing")
        st.markdown("---")

        # Navigation
        pages = [
            "Overview",
            "Study Area",
            "Data Quality",
            "Environmental Analysis",
            "Event Detection",
            "Susceptibility Prediction",
            "Feature Importance",
        ]

        if "selected_page" not in st.session_state or st.session_state["selected_page"] not in pages:
            st.session_state["selected_page"] = pages[0]

        selected_page = st.radio(
            "Select Section",
            pages,
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("---")
        st.markdown("**Filters**")

        # Year Range Filter
        years = st.slider(
            "Observation Window",
            min_value=2017,
            max_value=2020,
            value=(2017, 2020),
            step=1,
            help="Filter observations by event year (covers Sentinel active constellation baseline)",
        )

        # Geographic Subregion Filter
        regions = [
            "All Himalayan Regions",
            "India (J&K / Ladakh / HP)",
            "India (Uttarakhand)",
            "India (Northeast / Sikkim)",
            "Nepal",
        ]
        selected_region = st.selectbox(
            "Physiographic Subregion",
            regions,
            index=0,
            help="Filter spatial points by geographic subregion across the Himalayan mountain arc",
        )

        # Spatial Purge Distance Selection
        purge_options = [
            "100 km (Operational Standard)",
            "0 km (Unpurged Baseline)",
            "50 km",
            "200 km",
            "437.8 km (Semivariogram Range)",
        ]
        selected_purge = st.selectbox(
            "Spatial Outer CV Purge",
            purge_options,
            index=0,
            help="Outer test-to-train geographic exclusion buffer distance",
        )

        return {
            "page": selected_page,
            "years": years,
            "region": selected_region,
            "purge_distance": selected_purge,
        }
