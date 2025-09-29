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

# Newly added utilities
from utils.config import load_config
from utils.logging import get_logger
from utils.data_loader import load_demo_dataframe, analyze_demo_dataframe

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
        "analysis": None,  # AnalysisResult placeholder
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


def render_metrics_row(total_spent: float | None = None, num_tx: int | None = None, num_cats: int | None = None) -> None:
    """Render a responsive row of financial metric cards with optional values."""
    col1, col2, col3 = st.columns(3)
    with col1:
        if total_spent is None:
            st.metric(label="Total Spent", value=f"{DEFAULT_CURRENCY}0")
        else:
            st.metric(label="Total Spent", value=f"{DEFAULT_CURRENCY}{total_spent:,.2f}")
    with col2:
        st.metric(label="Transactions", value=str(num_tx or 0))
    with col3:
        st.metric(label="Categories", value=str(num_cats or 0))


def render_summary_section() -> None:
    """Render the summary statistics section, using demo data if available."""
    with st.container(border=True):
        st.markdown("### Summary Statistics 📊")

        analysis = st.session_state.get("analysis")
        if analysis is None:
            st.write(
                "Upload your data or load demo data to see personalized summaries, trends, and spending breakdowns."
            )
            render_metrics_row()
        else:
            render_metrics_row(
                total_spent=analysis.total_spent,
                num_tx=analysis.num_transactions,
                num_cats=len(analysis.totals_by_category),
            )


def render_visualizations_section() -> None:
    """Render the visualizations placeholder section with a minimal chart."""
    with st.container(border=True):
        st.markdown("### Visualizations 🎯")
        try:
            df = st.session_state.get("df")
            if df is None or df.empty:
                df_placeholder = pd.DataFrame({"Day": [], "Amount": []})
                fig = px.line(df_placeholder, x="Day", y="Amount", title="Spending Over Time (placeholder)")
            else:
                df_plot = df.copy()
                df_plot["Day"] = df_plot["timestamp"].dt.date
                df_plot = df_plot.groupby("Day")["amount"].sum().reset_index(name="Amount")
                fig = px.line(df_plot, x="Day", y="Amount", title="Spending Over Time (demo)")
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
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


def maybe_load_demo_data() -> None:
    """Populate session with demo data and analysis if Demo Data is selected."""
    if st.session_state.get("data_source") == "Demo Data":
        df = load_demo_dataframe()
        st.session_state["df"] = df
        st.session_state["analysis"] = analyze_demo_dataframe(df)


def main() -> None:
    """Application entry point."""
    set_page_config()

    # Init
    config = load_config()
    logger = get_logger()
    init_session_state()

    render_header()
    render_sidebar()

    # Data layer
    maybe_load_demo_data()

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
