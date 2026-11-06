"""Sidebar navigation and global filter panel."""

import streamlit as st
from typing import Dict, Any, List
from dashboard.config import ASSETS_DIR, THEME_COLORS


def render_sidebar() -> Dict[str, Any]:
    """Renders the dark navy sidebar with navigation, global filters, and metadata.

    Returns:
        Dictionary containing selected navigation page and active filter values.
    """
    with st.sidebar:
        # Logo & Header
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=70)

        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {THEME_COLORS['primary']}; text-transform: uppercase; letter-spacing: 0.08em;">
                    GEOSPATIAL INTELLIGENCE
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;">
                    HIMALAYAN LANDSLIDE
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">
                    Detection & Susceptibility Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 0.8rem 0;'>", unsafe_allow_html=True)

        # Page Navigation
        pages = [
            "Executive Overview",
            "Geographic Explorer",
            "Landslide Data Explorer",
            "Environmental Analysis",
            "Machine Learning Performance",
            "Susceptibility Analysis",
            "Data Quality & Methodology",
        ]

        page_icons = {
            "Executive Overview": "📊",
            "Geographic Explorer": "🗺️",
            "Landslide Data Explorer": "📋",
            "Environmental Analysis": "🛰️",
            "Machine Learning Performance": "⚡",
            "Susceptibility Analysis": "🔥",
            "Data Quality & Methodology": "🛡️",
        }

        # Initialize session state for page if not set
        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = pages[0]

        selected_page = st.radio(
            "Navigation",
            pages,
            format_func=lambda p: f"{page_icons.get(p, '•')}  {p}",
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.2rem 0 0.8rem 0;'>", unsafe_allow_html=True)

        # Global Filtering Section
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; font-weight: 700; color: {THEME_COLORS['secondary']}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;">
                GLOBAL FILTERS
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Year Range
        years = st.slider(
            "Event Year Range",
            min_value=2017,
            max_value=2020,
            value=(2017, 2020),
            step=1,
            help="Filter observations by event year (covers Sentinel-1/2 baseline)",
        )

        # Country / Subregion Filter
        regions = [
            "All Himalayan Regions",
            "India (J&K / Ladakh / HP)",
            "India (Uttarakhand)",
            "India (Northeast / Sikkim)",
            "Nepal",
            "Bhutan",
            "Pakistan / PoK",
            "China (Tibet)",
        ]
        selected_region = st.selectbox(
            "Geographic Region",
            regions,
            index=0,
            help="Filter observations by geographic sub-basin / political division",
        )

        # Minimum Susceptibility Threshold for alerts
        min_susceptibility = st.slider(
            "Susceptibility Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Highlight grid cells exceeding this ensemble susceptibility threshold",
        )

        if st.button("Reset All Filters", use_container_width=True):
            st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.2rem 0 0.8rem 0;'>", unsafe_allow_html=True)

        # System Provenance Pill
        st.markdown(
            """
            <div style="background: rgba(28, 37, 65, 0.7); border: 1px solid #2A3B5C; border-radius: 8px; padding: 0.75rem; font-size: 0.72rem; color: #94A3B8;">
                <div style="color: #48E5C2; font-weight: 600; margin-bottom: 4px;">SYSTEM STATUS</div>
                <div>• Observations: <b>254 points (1:1)</b></div>
                <div>• Susceptibility Grid: <b>0.5° (~55 km)</b></div>
                <div>• Models: <b>RF & XGBoost</b></div>
                <div>• Spatial CV: <b>5-Fold Clustered</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "page": selected_page,
        "year_range": years,
        "region": selected_region,
        "min_susceptibility": min_susceptibility,
    }

