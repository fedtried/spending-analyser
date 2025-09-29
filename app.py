"""AI Spending Analyser - Prototype

A professional, deployable Streamlit application foundation for financial analytics.

- Wide layout, custom page config, and dark theme (via .streamlit/config.toml)
- Polished header, sidebar, main content placeholders, and footer
- Strong UX with responsive columns, containers, and expanders
- Ready for Streamlit Cloud deployment

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px

# Constants
APP_NAME: str = "AI Spending Analyser"
TAGLINE: str = "Understand your financial habits with visual analytics and AI insights"
DEFAULT_CURRENCY: str = "£"


def set_page_config() -> None:
    """Configure Streamlit page settings early to avoid layout shifts."""
    st.set_page_config(
        page_title=f"{APP_NAME} · Financial Analytics",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    """Initialize Streamlit session state variables used across the app."""
    defaults = {
        "data_source": "Demo Data",  # or "Upload CSV"
        "analysis_period": "Last 30 days",
        "view_option": "Overview",
        "currency": DEFAULT_CURRENCY,
        "df": None,  # placeholder for DataFrame when added later
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Render the application header with title and tagline."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:1.8rem">💰</div>
            <div>
                <div style="font-size:1.6rem; font-weight:700; letter-spacing:0.2px;">{APP_NAME}</div>
                <div style="opacity:0.8; margin-top:2px;">{TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_sidebar() -> None:
    """Render the sidebar controls for data source, period, and view options."""
    with st.sidebar:
        st.markdown("### Settings")

        st.markdown("**Data Source**")
        data_source = st.radio(
            label="Choose data source",
            options=["Demo Data", "Upload CSV"],
            index=0 if st.session_state["data_source"] == "Demo Data" else 1,
            horizontal=True,
            key="data_source",
            help="Load prebuilt demo data or upload your own CSV.",
        )

        st.markdown("**Analysis Period**")
        period = st.selectbox(
            label="Time range",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date"],
            index=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date"].index(
                st.session_state["analysis_period"]
            ),
            key="analysis_period",
        )

        st.markdown("**View Options**")
        view_option = st.selectbox(
            label="Primary view",
            options=["Overview", "Categories", "Merchants"],
            index=["Overview", "Categories", "Merchants"].index(st.session_state["view_option"]),
            key="view_option",
        )

        st.caption("Use the controls above to configure your analysis.")

        with st.expander("About this prototype", expanded=False):
            st.write(
                "This is a prototype. Data upload, AI insights, and advanced filters will be added incrementally."
            )


def render_metrics_row() -> None:
    """Render a responsive row of financial metric cards with placeholder values."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Spent", value=f"{DEFAULT_CURRENCY}0")
    with col2:
        st.metric(label="Transactions", value="0")
    with col3:
        st.metric(label="Categories", value="0")


def render_summary_section() -> None:
    """Render the summary statistics placeholder section."""
    with st.container(border=True):
        st.markdown("### Summary Statistics 📊")
        st.write(
            "Upload your data or load demo data to see personalized summaries, trends, and spending breakdowns."
        )
        render_metrics_row()


def render_visualizations_section() -> None:
    """Render the visualizations placeholder section with an empty state chart."""
    with st.container(border=True):
        st.markdown("### Visualizations 🎯")
        # Placeholder empty chart using a minimal DataFrame
        try:
            df_placeholder = pd.DataFrame({"Day": [], "Amount": []})
            fig = px.line(df_placeholder, x="Day", y="Amount", title="Spending Over Time (placeholder)")
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # graceful handling for any rendering issues
            st.info("Visualization will appear here once data is available.")
            st.caption(f"Renderer note: {exc}")


def render_ai_insights_section() -> None:
    """Render the AI insights placeholder with a spinner mock-up."""
    with st.container(border=True):
        st.markdown("### AI Insights 🤖")
        with st.spinner("Analyzing spending patterns with AI (placeholder)..."):
            st.write(
                "AI-powered insights will summarize key behaviors, detect anomalies, and suggest budget improvements."
            )
            st.write("Connect your API key in `/.streamlit/secrets.toml` when enabling AI features.")


def render_footer() -> None:
    """Render a subtle footer."""
    st.divider()
    st.caption(
        "Built with Streamlit. Prototype for demonstration purposes."
    )


def render_empty_state() -> None:
    """Render the empty state message guiding users to load or upload data."""
    st.info("Upload your data or load demo data to begin")


def main() -> None:
    """Application entry point."""
    set_page_config()
    init_session_state()

    render_header()
    render_sidebar()

    # Main content layout
    welcome = st.container()
    with welcome:
        st.success(
            "Welcome to AI Spending Analyser — a modern dashboard to explore your finances with clarity and insight."
        )

    # Content areas in responsive layout
    summary_area = st.container()
    with summary_area:
        render_summary_section()

    viz_area = st.container()
    with viz_area:
        render_visualizations_section()

    ai_area = st.container()
    with ai_area:
        render_ai_insights_section()

    # Empty state prompt
    render_empty_state()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
