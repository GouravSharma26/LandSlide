"""Sidebar navigation and global research filter panel."""

import streamlit as st
from typing import Dict, Any
from dashboard.config import ASSETS_DIR, THEME_COLORS


def render_sidebar() -> Dict[str, Any]:
    """Renders the dark navy sidebar with 8-section research navigation, filters, and provenance badges.

    Returns:
        Dictionary containing selected navigation page and active filter values.
    """
    with st.sidebar:
        # Header / Branding
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=65)

        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <div style="font-size: 0.68rem; font-weight: 700; color: {THEME_COLORS['primary']}; text-transform: uppercase; letter-spacing: 0.09em;">
                    GEOSPATIAL AI RESEARCH
                </div>
                <div style="font-size: 1.12rem; font-weight: 800; color: #FFFFFF; line-height: 1.25;">
                    HIMALAYAN LANDSLIDE
                </div>
                <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 2px;">
                    Event Detection & Susceptibility Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 0.6rem 0;'>", unsafe_allow_html=True)

        # 8 Research Sections Navigation
        pages = [
            "Executive Overview",
            "Geographic Explorer",
            "Data Catalog & Quality",
            "Environmental Analysis",
            "Event Detection",
            "Susceptibility Prediction",
            "Explainability & Uncertainty",
            "Research Methodology & Governance",
        ]

        page_icons = {
            "Executive Overview": "📊",
            "Geographic Explorer": "🗺️",
            "Data Catalog & Quality": "🛡️",
            "Environmental Analysis": "🛰️",
            "Event Detection": "⚡",
            "Susceptibility Prediction": "🔥",
            "Explainability & Uncertainty": "🧠",
            "Research Methodology & Governance": "📜",
        }

        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = pages[0]

        selected_page = st.radio(
            "Navigation",
            pages,
            format_func=lambda p: f"{page_icons.get(p, '•')}  {p}",
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.0rem 0 0.6rem 0;'>", unsafe_allow_html=True)

        # Global Filtering Section
        st.markdown(
            f"""
            <div style="font-size: 0.70rem; font-weight: 700; color: {THEME_COLORS['secondary']}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                RESEARCH DOMAIN FILTERS
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        purge_options = ["100 km (Operational Standard)", "0 km (Unpurged Baseline)", "50 km", "200 km", "437.8 km (Semivariogram Range)"]
        selected_purge = st.selectbox(
            "Spatial Outer CV Purge",
            purge_options,
            index=0,
            help="Outer test-to-train geographic exclusion buffer distance",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.0rem 0 0.6rem 0;'>", unsafe_allow_html=True)

        # Scientific Provenance Badges
        st.markdown(
            f"""
            <div style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
                PROVENANCE STATUS
            </div>
            <div style="background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38BDF8; padding: 5px 8px; margin-bottom: 6px; border-radius: 2px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #38BDF8;">Version 1.0</div>
                <div style="font-size: 0.66rem; color: #CBD5E1;">Historical Diagnostic Baseline (N=254)</div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #F59E0B; padding: 5px 8px; margin-bottom: 6px; border-radius: 2px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #F59E0B;">Version 2.0</div>
                <div style="font-size: 0.66rem; color: #CBD5E1;">EXTRACTION_PENDING_COLAB_GEE</div>
            </div>
            <div style="background: rgba(168, 85, 247, 0.1); border-left: 3px solid #A855F7; padding: 5px 8px; margin-bottom: 6px; border-radius: 2px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #A855F7;">Workflow A (Detection)</div>
                <div style="font-size: 0.66rem; color: #CBD5E1;">Experimental Point-Zone Disturbance</div>
            </div>
            <div style="background: rgba(0, 240, 255, 0.1); border-left: 3px solid #00F0FF; padding: 5px 8px; margin-bottom: 6px; border-radius: 2px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #00F0FF;">Workflow B (Susceptibility)</div>
                <div style="font-size: 0.66rem; color: #CBD5E1;">Rainfall-Conditioned Diagnostic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "page": selected_page,
            "years": years,
            "region": selected_region,
            "purge_distance": selected_purge,
        }
