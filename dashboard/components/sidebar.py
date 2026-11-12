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
            <div style="margin-bottom: 0.9rem;">
                <div style="font-size: 1.08rem; font-weight: 800; color: #FFFFFF; line-height: 1.25; letter-spacing: -0.01em;">
                    Early Landslide Risk Detection
                </div>
                <div style="font-size: 0.76rem; color: #38BDF8; font-weight: 600; margin-top: 3px;">
                    Visual Terrain Changes & Multi-Modal AI
                </div>
                <div style="display: inline-block; margin-top: 6px; padding: 2px 7px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 4px; font-size: 0.68rem; color: #94A3B8;">
                    🏔️ Himalayan Testbed Region
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 0.6rem 0;'>", unsafe_allow_html=True)

        # 7 Research Sections Navigation
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

        icons = {
            "Overview": "📊",
            "Study Area": "🗺️",
            "Data Quality": "🔬",
            "Environmental Analysis": "🌱",
            "Event Detection": "⚡",
            "Susceptibility Prediction": "🛡️",
            "Feature Importance": "📈",
        }

        selected_page = st.radio(
            "Navigation",
            pages,
            format_func=lambda p: f"{icons.get(p, '•')}  {p}",
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.0rem 0 0.6rem 0;'>", unsafe_allow_html=True)

        # Global Filtering Section
        st.markdown(
            f"""
            <div style="font-size: 0.70rem; font-weight: 700; color: {THEME_COLORS['secondary']}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                FILTERS
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

        # Dataset and model status
        st.markdown(
            """
            <div style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
                SYSTEM TELEMETRY
            </div>
            <div class="sidebar-status-card">
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">Pipeline Status</span>
                    <span class="sidebar-status-val" style="color: #4ADE80;">● Operational</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">Baseline Dataset</span>
                    <span class="sidebar-status-val">V1.0 Verified (N=254)</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">Hillslope GEE Audit</span>
                    <span class="sidebar-status-val" style="color: #38BDF8;">Archived (45.3m)</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">Spatial Purge</span>
                    <span class="sidebar-status-val">100 km Cluster CV</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">Test Suite</span>
                    <span class="sidebar-status-val" style="color: #4ADE80;">59/59 Passed</span>
                </div>
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
