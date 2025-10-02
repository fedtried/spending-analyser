from __future__ import annotations

import os
import io
import time
from typing import List, Optional
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import load_config
from utils.logging import get_logger
from utils.data_loader import (
    load_demo_dataframe,
    load_user_dataframe,
    analyze_dataframe,
)
from utils.gemini_streaming import create_streaming_processor
from components.chat_interface import ChatInterface


def get_accessible_colors():
    """Return a palette of accessible colors that work well together and are colorblind-friendly."""
    return {
        'income': 'rgba(76, 175, 80, 0.7)',      # Accessible green
        'spending': 'rgba(244, 67, 54, 0.7)',    # Accessible red
        'balance': 'rgba(33, 150, 243, 1)',      # Accessible blue
        'grid': 'rgba(128, 128, 128, 0.4)',     # Semi-transparent gray
        'primary': '#4CAF50',                    # Matches theme primary
        'secondary': 'rgba(255, 193, 7, 0.8)',  # Accessible amber
        'tertiary': 'rgba(156, 39, 176, 0.7)',  # Accessible purple
    }

APP_NAME: str = "Spending Analyst"
TAGLINE: str = "Your financial companion that gets it"
DEFAULT_CURRENCY: str = "£"

# Handle different Gemini SDK versions
try:  # Preferred modern client
    from google import genai as genai_client  # type: ignore
except Exception:  # Fallback to legacy namespace
    genai_client = None  # type: ignore

try:
    import google.generativeai as genai_legacy  # type: ignore
except Exception:  # pragma: no cover
    genai_legacy = None  # type: ignore


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
    from datetime import date
    defaults = {
        "data_source": "Demo Data",
        "analysis_period": "Custom Date Range",
        "currency": DEFAULT_CURRENCY,
        "df": None,
        "df_raw": None,
        "analysis": None,
        "ai_pdf_summary": None,
        "chat_interface": None,
        "streaming_processor": None,
        "demo_ai_processed": False,
        "processing_state": "idle",  # idle, streaming, complete
        "is_processing": False,
        "stream_buffer": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_chat_components() -> None:
    """Initialize chat interface and streaming processor if not already done."""
    if st.session_state.get("chat_interface") is None:
        st.session_state["chat_interface"] = ChatInterface()
    
    if st.session_state.get("streaming_processor") is None:
        try:
            st.session_state["streaming_processor"] = create_streaming_processor()
        except Exception as e:
            st.error(f"Failed to initialize streaming processor: {str(e)}")
            st.session_state["streaming_processor"] = None


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


def render_data_source_section() -> None:
    """Render the data source selection section with demo data showcase."""
    # Only show this section if we haven't started processing
    if (st.session_state.get("processing_state") == "idle" and 
        not st.session_state.get("demo_ai_processed")):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # st.markdown("### Demo Data Analysis")
                st.markdown("###  📊 Experience AI-powered financial insights with a sample dataset")
                st.caption("💡 **Tip:** This demo uses realistic UK bank statement data. All personal information has been anonymized for demonstration purposes.")
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "🚀 Start AI Analysis", 
                    type="primary", 
                    use_container_width=True,
                    help="Begin analyzing the demo data with AI insights"
                ):
                    chat_interface = st.session_state.get("chat_interface")
                    if chat_interface:
                        chat_interface.clear_messages()
                    
                    st.session_state["demo_ai_processed"] = False
                    st.session_state["is_processing"] = False
                    st.session_state["stream_buffer"] = ""
                    st.session_state["processing_state"] = "streaming"
                    st.session_state["data_source"] = "Demo Data"
                    
                    df = load_demo_dataframe()
                    st.session_state["df_raw"] = df
                    st.session_state["df"] = df
                    st.session_state["analysis"] = analyze_dataframe(df)
                    
                    st.rerun()


def render_sidebar() -> None:
    """Render the sidebar controls for analysis period and view options."""
    with st.sidebar:
        st.markdown("### Settings")
        
        st.markdown("**Analysis Period**")
        date_range_type = st.selectbox(
            label="Time range type",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date", "Custom Date Range"],
            index=4,
            key="date_range_type",
        )
        
        from datetime import timedelta
        today = date.today()
        
        if date_range_type == "Custom Date Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=date(2025, 7, 1),
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=date(2025, 9, 30),
                    key="end_date"
                )
            
            if start_date and end_date and start_date > end_date:
                st.error("Start date must be before end date")
        else:
            if date_range_type == "Last 7 days":
                start_date = today - timedelta(days=6)
                end_date = today
            elif date_range_type == "Last 30 days":
                start_date = today - timedelta(days=29)
                end_date = today
            elif date_range_type == "Last 90 days":
                start_date = today - timedelta(days=89)
                end_date = today
            elif date_range_type == "Year to date":
                start_date = date(today.year, 1, 1)
                end_date = today
            
            st.caption(f"Selected: {start_date} to {end_date}")
            st.session_state["start_date"] = start_date
            st.session_state["end_date"] = end_date
        
        st.markdown("---")
        st.markdown("### Reset")
        if st.button("🔄 Start Over", use_container_width=True, help="Clear analysis and start fresh"):
            for key in ["df", "df_raw", "analysis", "ai_pdf_summary", "demo_ai_processed", 
                       "is_processing", "stream_buffer"]:
                if key in st.session_state:
                    st.session_state[key] = None if key not in ["demo_ai_processed", "is_processing"] else False
            
            st.session_state["processing_state"] = "idle"
            st.session_state["stream_buffer"] = ""
            
            chat_interface = st.session_state.get("chat_interface")
            if chat_interface:
                chat_interface.clear_messages()
            
            st.rerun()


def render_metrics_row(total_spent: float | None = None, total_income: float | None = None, total_net: float | None = None) -> None:
    """Render a responsive row of financial metric cards with optional values."""
    col1, col2, col3 = st.columns(3)
    with col1:
        if total_spent is None:
            st.metric(label="Total Spent", value=f"{DEFAULT_CURRENCY}0")
        else:
            st.metric(label="Total Spent", value=f"{DEFAULT_CURRENCY}{total_spent:,.2f}")
    with col2:
        if total_income is None:
            st.metric(label="Total Income", value=f"{DEFAULT_CURRENCY}0")
        else:
            st.metric(label="Total Income", value=f"{DEFAULT_CURRENCY}{total_income:,.2f}")
    with col3:
        if total_net is None:
            st.metric(label="Total Net", value=f"{DEFAULT_CURRENCY}0")
        else:
            st.metric(label="Total Net", value=f"{DEFAULT_CURRENCY}{total_net:,.2f}")


def render_chat_section() -> None:
    """Render the chat interface section with TRUE streaming."""
    if st.session_state.get("processing_state") != "idle":
        chat_interface = st.session_state.get("chat_interface")
        streaming_processor = st.session_state.get("streaming_processor")
        
        if chat_interface and streaming_processor:
            with st.container(border=True):
                
                if (st.session_state.get("processing_state") == "streaming" and 
                    not st.session_state.get("is_processing")):
                    
                    st.session_state["is_processing"] = True
                    df = st.session_state.get("df_raw")
                    
                    if df is not None:
                        chat_placeholder = st.empty()
                        
                        # Show initial demo analysis messages immediately
                        chat_interface.start_demo_analysis()
                        with chat_placeholder.container():
                            chat_interface.render_chat_container()
                        
                        try:
                            # Stream chunks directly as they arrive (no pre-buffer)
                            full_text = ""
                            for chunk in streaming_processor.process_demo_data_streaming(df, chat_interface):
                                full_text += chunk
                                chat_interface.update_assistant_message(chunk, append=True)
                                
                                with chat_placeholder.container():
                                    chat_interface.render_chat_container()
                                
                                # Control streaming speed based on chunk size
                                if len(chunk) < 10:
                                    time.sleep(0.03)
                                elif len(chunk) < 50:
                                    time.sleep(0.05)
                                else:
                                    time.sleep(0.1)
                            
                            st.session_state["ai_pdf_summary"] = full_text
                            st.session_state["demo_ai_processed"] = True
                            st.session_state["processing_state"] = "complete"
                            chat_interface.complete_processing(df)
                            
                            with chat_placeholder.container():
                                chat_interface.render_chat_container()
                            
                        except Exception as e:
                            st.error(f"Streaming failed: {str(e)}")
                            chat_interface.add_message("system", f"⌛ Analysis failed: {str(e)}")
                        finally:
                            st.session_state["is_processing"] = False
                
                else:
                    chat_interface.render_chat_container()


def render_visualizations_section() -> None:
    """Render the visualizations section."""
    if st.session_state.get("processing_state") != "idle":
        with st.container(border=True):
            st.markdown("### Visualizations 🎯")
            
            df_raw = st.session_state.get("df_raw")
            df = df_raw if df_raw is not None else st.session_state.get("df")
            
            if df is None or df.empty:
                if st.session_state.get("processing_state") == "streaming":
                    st.info("📊 Visualizations will appear here once analysis is complete...")
                else:
                    st.info("No data available for visualization.")
                return
            
            try:
                
                start_date, end_date = get_current_date_range()
                if start_date and end_date:
                    st.caption(f"Analysis period: {start_date} to {end_date}")
                    df = filter_dataframe_by_date_range(df, start_date, end_date)

                if df is not None and not df.empty:
                    local_analysis = analyze_dataframe(df)
                    render_metrics_row(
                        total_spent=local_analysis.total_spent,
                        total_income=local_analysis.total_income,
                        total_net=local_analysis.total_net,
                    )
                else:
                    render_metrics_row()

                df_bal = df.copy()
                df_bal["Day"] = df_bal["timestamp"].dt.date
                
                daily_income = df_bal[df_bal["amount"] > 0].groupby("Day")["amount"].sum().reset_index(name="Income")
                daily_spending = df_bal[df_bal["amount"] < 0].groupby("Day")["amount"].sum().reset_index(name="Spending")
                daily_spending["Spending"] = -daily_spending["Spending"]
                
                daily_bal = pd.merge(daily_income, daily_spending, on="Day", how="outer").fillna(0)
                
                if "balance_after" in df_bal.columns:
                    daily_balance = df_bal.groupby("Day")["balance_after"].last().reset_index()
                    daily_balance.columns = ["Day", "Balance"]
                    daily_bal = pd.merge(daily_bal, daily_balance, on="Day", how="outer").fillna(0)
                else:
                    daily_bal["Daily_Net"] = daily_bal["Income"] - daily_bal["Spending"]
                    daily_bal["Balance"] = daily_bal["Daily_Net"].cumsum()
                
                colors = get_accessible_colors()
                
                fig = make_subplots(
                    rows=1, cols=1,
                    specs=[[{"secondary_y": True}]],
                    subplot_titles=("Daily Income vs Spending with Balance Trend",)
                )
                
                fig.add_trace(
                    go.Bar(x=daily_bal["Day"], y=daily_bal["Income"], 
                           name="Income", marker_color=colors['income'],
                           hovertemplate="<b>%{x}</b><br>Income: £%{y:,.0f}<extra></extra>"),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Bar(x=daily_bal["Day"], y=daily_bal["Spending"], 
                           name="Spending", marker_color=colors['spending'],
                           hovertemplate="<b>%{x}</b><br>Spending: £%{y:,.0f}<extra></extra>"),
                    secondary_y=False,
                )
                
                daily_bal_unique = daily_bal.drop_duplicates(subset=['Day'], keep='last').sort_values('Day')
                
                fig.add_trace(
                    go.Scatter(x=daily_bal_unique["Day"], y=daily_bal_unique["Balance"], 
                              name="Account Balance", mode="lines+markers", 
                              line=dict(color=colors['balance'], width=3),
                              marker=dict(size=5, color=colors['balance'], symbol="circle"),
                              hovertemplate="<b>%{x}</b><br>Balance: £%{y:,.0f}<extra></extra>"),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Amount (£)", secondary_y=False)
                fig.update_yaxes(title_text="Balance (£)", secondary_y=True)
                
                fig.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=60, b=20),
                    barmode="group",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'], secondary_y=False)
                fig.update_yaxes(showgrid=False, secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Category Spend Breakdown")
                
                spend_df = df[df["amount"] < 0].copy()
                if not spend_df.empty:
                    spend_df["Spend"] = -spend_df["amount"]
                    cat_totals = (
                        spend_df.groupby(spend_df["category"].fillna("Uncategorized"))["Spend"]
                            .sum().sort_values(ascending=False)
                    )
                    cat_df = cat_totals.reset_index()
                    cat_df.columns = ["Category", "Spend"]
                    
                    left_col, right_col = st.columns([1, 2])
                    
                    with left_col:
                        st.markdown("**Category Statistics**")
                        
                        st.metric(
                            label="Total Categories", 
                            value=len(cat_df),
                            help="Number of different spending categories"
                        )
                        
                        if len(cat_df) > 0:
                            top_category = cat_df.iloc[0]
                            st.metric(
                                label="Top Category", 
                                value=top_category["Category"],
                                delta=f"£{top_category['Spend']:,.0f}",
                                help="Category with highest spending"
                            )
                            
                            top_spend = cat_df.iloc[0]["Spend"]
                            total_spend = cat_df["Spend"].sum()
                            top_percentage = (top_spend / total_spend) * 100
                            st.metric(
                                label="Top Category %", 
                                value=f"{top_percentage:.1f}%",
                                help="Percentage of total spending in top category"
                            )
                    
                    with right_col:
                        # Create accessible color palette for pie chart
                        accessible_palette = [
                            colors['primary'], colors['secondary'], colors['tertiary'],
                            colors['income'], colors['spending'], colors['balance'],
                            'rgba(255, 152, 0, 0.8)',  # Orange
                            'rgba(0, 150, 136, 0.8)',  # Teal
                            'rgba(121, 85, 72, 0.8)',  # Brown
                            'rgba(63, 81, 181, 0.8)'   # Indigo
                        ]
                        
                        cat_fig = px.pie(cat_df, names="Category", values="Spend", 
                                        title="Share of Spending by Category", hole=0.4,
                                        color_discrete_sequence=accessible_palette)
                        cat_fig.update_traces(textposition="inside", textinfo="percent+label")
                        cat_fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(cat_fig, use_container_width=True)
                else:
                    st.info("No spending transactions to build a category breakdown.")
                    
            except Exception as exc:
                if st.session_state.get("processing_state") == "streaming":
                    st.info("📊 Processing data for visualizations...")
                else:
                    st.error(f"Error rendering visualizations: {exc}")


def filter_dataframe_by_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter dataframe by the selected date range."""
    if df is None or df.empty:
        return df
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
    return filtered_df


def get_current_date_range():
    """Get the current date range from session state or widgets."""
    date_range_type = st.session_state.get("date_range_type", "Custom Date Range")
    
    start_date = st.session_state.get("start_date", date(2025, 7, 1))
    end_date = st.session_state.get("end_date", date(2025, 9, 30))
    return start_date, end_date


def render_footer() -> None:
    """Render a subtle footer."""
    st.divider()
    st.caption("Built with Streamlit.")


def main() -> None:
    """Application entry point."""
    set_page_config()

    config = load_config()
    logger = get_logger()
    init_session_state()
    init_chat_components()

    render_header()
    render_sidebar()
    render_data_source_section()
    render_chat_section()
    render_visualizations_section()
    render_footer()


if __name__ == "__main__":
    main()