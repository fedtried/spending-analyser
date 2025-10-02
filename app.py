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
import io
import time
from typing import List, Optional
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Newly added utilities
from utils.config import load_config
from utils.logging import get_logger
from utils.data_loader import (
    load_demo_dataframe,
    load_user_dataframe,
    analyze_dataframe,
)
from utils.gemini_streaming import create_streaming_processor
from components.chat_interface import ChatInterface

# Constants
APP_NAME: str = "Spending Analyst"
TAGLINE: str = "Your financial companion that gets it - no judgment, just insights that actually help"
DEFAULT_CURRENCY: str = "£"

# Optional Gemini imports (SDK may expose either interface depending on version)
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
        "app_state": "initial",  # "initial" or "final" - controls UI visibility
        "data_source": "Demo Data",  # Will be determined by file upload
        "analysis_period": "Custom Date Range",
        # "date_range_type": "Custom Date Range",  # Removed to avoid conflict with widget default
        # "start_date": date(2025, 7, 1),  # Removed to avoid conflict with widget default
        # "end_date": date(2025, 9, 30),   # Removed to avoid conflict with widget default
        "currency": DEFAULT_CURRENCY,
        "df": None,  # placeholder for DataFrame when added later
        "analysis": None,  # AnalysisResult placeholder
        "ai_pdf_summary": None,  # Stores one-sentence summary 
        # Chat interface state
        "chat_interface": None,  # ChatInterface instance
        "streaming_processor": None,  # StreamingGeminiProcessor instance
        "demo_ai_processed": False,  # Flag to track if demo data has been AI processed
        "processing_state": "idle",  # idle, uploading, streaming, complete
        "start_pdf_processing": False,  # Flag to start PDF processing
        "start_demo_processing": False,  # Flag to start demo processing
        "is_processing": False,  # Guard to prevent duplicate runs
        "last_filtered_start_date": None,  # Track last filtered start date
        "last_filtered_end_date": None,  # Track last filtered end date
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
    # Only show in initial state
    if st.session_state.get("app_state") != "initial":
        return
    
    # Main container with enhanced styling
    with st.container(border=True):
        # Header section with button on the right
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📊 Demo Data Analysis")
            st.markdown("**Experience AI-powered financial insights with our sample dataset**")
            st.caption("💡 **Tip:** This demo uses realistic UK bank statement data. All personal information has been anonymized for demonstration purposes.")
        
        with col2:
            # Add some vertical spacing to center the button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "🚀 Start AI Analysis", 
                type="primary", 
                use_container_width=True,
                help="Begin analyzing the demo data with AI insights"
            ):
                # Set flag to start demo processing
                st.session_state["start_demo_processing"] = True
                st.session_state["processing_state"] = "uploading"
                st.session_state["data_source"] = "Demo Data"
                # Change app state to final
                st.session_state["app_state"] = "final"
                st.rerun()


def render_sidebar() -> None:
    """Render the sidebar controls for analysis period and view options."""
    with st.sidebar:
        st.markdown("### Settings")

        st.caption("Configure your analysis period and view options.")

        st.markdown("**Analysis Period**")
        date_range_type = st.selectbox(
            label="Time range type",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date", "Custom Date Range"],
            index=4,  # Default to "Custom Date Range"
            key="date_range_type",
        )
        
        # Calculate date range based on selection
        from datetime import timedelta
        today = date.today()
        
        if date_range_type == "Custom Date Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=date(2025, 7, 1),  # Default value only
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=date(2025, 9, 30),  # Default value only
                    key="end_date"
                )
            
            # Validate date range
            if start_date and end_date and start_date > end_date:
                st.error("Start date must be before end date")
            else:
                # Check if dates have changed and trigger rerun
                old_start = st.session_state.get("start_date")
                old_end = st.session_state.get("end_date")
                if (old_start != start_date or old_end != end_date):
                    st.session_state["start_date"] = start_date
                    st.session_state["end_date"] = end_date
                    # Trigger rerun to reload data with new filter
                    st.rerun()
        else:
            # Calculate date range for preset options
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
        
        # Store the calculated dates in session state (only if not custom range to avoid conflicts)
        if date_range_type != "Custom Date Range":
            if (st.session_state.get("start_date") != start_date or 
                st.session_state.get("end_date") != end_date):
                st.session_state["start_date"] = start_date
                st.session_state["end_date"] = end_date
                # Trigger rerun to reload data with new filter
                st.rerun()
        
        # Show reset button in final state
        if st.session_state.get("app_state") == "final":
            st.markdown("---")
            st.markdown("### Reset")
            if st.button("🔄 Start Over", use_container_width=True, help="Clear analysis and start fresh"):
                # Clear all data and reset state
                st.session_state["df"] = None
                st.session_state["analysis"] = None
                st.session_state["ai_pdf_summary"] = None
                st.session_state["demo_ai_processed"] = False
                st.session_state["processing_state"] = "idle"
                st.session_state["uploaded_file"] = None
                st.session_state["data_source"] = "Demo Data"
                st.session_state["start_pdf_processing"] = False
                st.session_state["start_demo_processing"] = False
                st.session_state["is_processing"] = False
                st.session_state["last_filtered_start_date"] = None
                st.session_state["last_filtered_end_date"] = None
                # Reset app state to initial
                st.session_state["app_state"] = "initial"
                
                # Clear chat messages
                chat_interface = st.session_state.get("chat_interface")
                if chat_interface:
                    chat_interface.clear_messages()
                
                st.rerun()

        # Show chat interface metrics if available
        chat_interface = st.session_state.get("chat_interface")
        if chat_interface and st.session_state.get("processing_state") in ["streaming", "complete"]:
            st.markdown("---")
            chat_interface.render_sidebar_metrics()


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

def render_visualizations_section() -> None:
    """Render the visualizations placeholder section with a minimal chart."""
    # Only show visualizations in final state
    if st.session_state.get("app_state") != "final":
        return
        
    with st.container(border=True):
        st.markdown("### Visualizations 🎯")
        
        try:
            df = st.session_state.get("df")

            if df is None or df.empty:
                st.info("Visualization will appear here once data is available.")
                return

            # Overview metrics section - show when data is available
            st.markdown("#### Overview 📊")
            
            # Show current date range
            start_date, end_date = get_current_date_range()
            if start_date and end_date:
                st.caption(f"Analysis period: {start_date} to {end_date}")

            analysis = st.session_state.get("analysis")
            if analysis is not None:
                render_metrics_row(
                    total_spent=analysis.total_spent,
                    total_income=analysis.total_income,
                    total_net=analysis.total_net,
                )
            else:
                render_metrics_row()
            
            st.markdown("---")  # Separator between overview and charts

            # Daily Income vs Spending with Balance Trend Overlay
            df_bal = df.copy()
            df_bal["Day"] = df_bal["timestamp"].dt.date
            
            # Calculate daily income and spending based on amount sign (Income=+, Spend=-)
            daily_income = df_bal[df_bal["amount"] > 0].groupby("Day")["amount"].sum().reset_index(name="Income")
            
            daily_spending = df_bal[df_bal["amount"] < 0].groupby("Day")["amount"].sum().reset_index(name="Spending")
            daily_spending["Spending"] = -daily_spending["Spending"]  # Convert to positive for display
            
            # Merge income and spending data
            daily_bal = pd.merge(daily_income, daily_spending, on="Day", how="outer").fillna(0)
            
            # Get actual balance data from Balance_After column
            if "balance_after" in df_bal.columns:
                # Take the last transaction balance for each day
                daily_balance = df_bal.groupby("Day")["balance_after"].last().reset_index()
                daily_balance.columns = ["Day", "Balance"]
                
                # Merge balance with income/spending data
                daily_bal = pd.merge(daily_bal, daily_balance, on="Day", how="outer").fillna(0)
            else:
                # Fallback: calculate running balance if balance_after not available
                daily_bal["Daily_Net"] = daily_bal["Income"] - daily_bal["Spending"]
                daily_bal["Balance"] = daily_bal["Daily_Net"].cumsum()
            
            # Create the chart with dual y-axes
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=("Daily Income vs Spending with Balance Trend",)
            )
            
            # Add bars for income and spending
            fig.add_trace(
                go.Bar(x=daily_bal["Day"], y=daily_bal["Income"], 
                       name="Income", marker_color="green", opacity=0.7,
                       hovertemplate="<b>%{x}</b><br>Income: £%{y:,.0f}<extra></extra>"),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(x=daily_bal["Day"], y=daily_bal["Spending"], 
                       name="Spending", marker_color="red", opacity=0.7,
                       hovertemplate="<b>%{x}</b><br>Spending: £%{y:,.0f}<extra></extra>"),
                secondary_y=False,
            )
            
            # Ensure we have only one balance value per day
            daily_bal_unique = daily_bal.drop_duplicates(subset=['Day'], keep='last').sort_values('Day')
            
            # Add line for actual balance on secondary y-axis
            fig.add_trace(
                go.Scatter(x=daily_bal_unique["Day"], y=daily_bal_unique["Balance"], 
                          name="Account Balance", mode="lines+markers", 
                          line=dict(color="darkblue", width=3),
                          marker=dict(size=5, color="darkblue", symbol="circle"),
                          hovertemplate="<b>%{x}</b><br>Balance: £%{y:,.0f}<extra></extra>"),
                secondary_y=True,
            )
            
            # Set x-axis title
            fig.update_xaxes(title_text="Date")
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Amount (£)", secondary_y=False)
            fig.update_yaxes(title_text="Balance (£)", secondary_y=True)
            
            # Update layout
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
            
            # Style the axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", secondary_y=False)
            fig.update_yaxes(showgrid=False, secondary_y=True)  # Don't show grid on right axis to avoid clutter
            
            st.plotly_chart(fig, use_container_width=True)

            # Category Spend Breakdown with controls
            st.markdown("#### Category Spend Breakdown")
            
            spend_df = df[df["amount"] < 0].copy()
            if not spend_df.empty:
                spend_df["Spend"] = -spend_df["amount"]
                cat_totals = (
                    spend_df.groupby(spend_df["category"].fillna("Uncategorized"))["Spend"].sum().sort_values(ascending=False)
                )
                cat_df = cat_totals.reset_index()
                cat_df.columns = ["Category", "Spend"]
                
                # Layout with metrics on left and chart on right
                left_col, right_col = st.columns([1, 2])
                
                with left_col:
                    st.markdown("**Category Statistics**")
                    
                    # Total Categories metric
                    st.metric(
                        label="Total Categories", 
                        value=len(cat_df),
                        help="Number of different spending categories"
                    )
                    
                    # Top Category metric
                    if len(cat_df) > 0:
                        top_category = cat_df.iloc[0]
                        st.metric(
                            label="Top Category", 
                            value=top_category["Category"],
                            delta=f"{DEFAULT_CURRENCY}{top_category['Spend']:,.0f}",
                            help="Category with highest spending"
                        )
                        
                        # Top Category percentage
                        top_spend = cat_df.iloc[0]["Spend"]
                        total_spend = cat_df["Spend"].sum()
                        top_percentage = (top_spend / total_spend) * 100
                        st.metric(
                            label="Top Category %", 
                            value=f"{top_percentage:.1f}%",
                            help="Percentage of total spending in top category"
                        )
                
                with right_col:
                    cat_fig = px.pie(cat_df, names="Category", values="Spend", title="Share of Spending by Category", hole=0.4)
                    cat_fig.update_traces(textposition="inside", textinfo="percent+label")
                    cat_fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(cat_fig, use_container_width=True)
            else:
                st.info("No spending transactions to build a category breakdown.")
                
        except Exception as exc:
            st.info("Visualization will appear here once data is available.")
            st.caption(f"Renderer note: {exc}")




def render_chat_section() -> None:
    """Render the chat interface section for data analysis."""
    # Only show chat in final state
    if st.session_state.get("app_state") != "final":
        return
        
    chat_interface = st.session_state.get("chat_interface")
    if chat_interface and st.session_state.get("data_source") == "Demo Data":
        with st.container(border=True):
            data_source = st.session_state.get("data_source")
            st.markdown("### 💬 Financial Analysis Chat")
            
            # Handle processing flags - PDF PROCESSING COMMENTED OUT
            # if st.session_state.get("start_pdf_processing") and not st.session_state.get("is_processing"):
            #     st.session_state["start_pdf_processing"] = False
            #     uploaded_file = st.session_state.get("uploaded_file")
            #     if uploaded_file is not None:
            #         st.session_state["is_processing"] = True
            #         st.session_state["processing_state"] = "streaming"
            #         process_pdf_with_streaming(uploaded_file)
            
            if st.session_state.get("start_demo_processing") and not st.session_state.get("is_processing"):
                st.session_state["start_demo_processing"] = False
                st.session_state["is_processing"] = True
                st.session_state["processing_state"] = "streaming"
                process_demo_data_with_ai()
            
            # Only render chat container when not processing AND we haven't completed processing yet
            is_processing = st.session_state.get("is_processing", False)
            has_completed_processing = st.session_state.get("demo_ai_processed", False)
            
            if not is_processing and not has_completed_processing:
                chat_interface.render_chat_container()
            
            # Show download section if processing is complete and we have data - COMMENTED OUT FOR DEMO ONLY
            # if st.session_state.get("data_source") == "Upload PDF":
            #     chat_interface.render_download_section()


def render_footer() -> None:
    """Render a subtle footer."""
    st.divider()
    st.caption(
        "Built with Streamlit."
    )


def render_empty_state() -> None:
    """Render the empty state message guiding users to load or upload data."""
    pass


def filter_dataframe_by_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter dataframe by the selected date range."""
    if df is None or df.empty:
        return df
    
    # Convert dates to datetime for comparison
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Filter the dataframe
    filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
    return filtered_df


def get_current_date_range():
    """Get the current date range from session state or widgets."""
    date_range_type = st.session_state.get("date_range_type", "Custom Date Range")
    
    if date_range_type == "Custom Date Range":
        # For custom date range, get from session state (set by the widgets)
        # Provide fallback defaults if not in session state
        start_date = st.session_state.get("start_date", date(2025, 7, 1))
        end_date = st.session_state.get("end_date", date(2025, 9, 30))
        return start_date, end_date
    else:
        # For preset ranges, get from session state
        start_date = st.session_state.get("start_date", date(2025, 7, 1))
        end_date = st.session_state.get("end_date", date(2025, 9, 30))
        return start_date, end_date


# PDF PROCESSING FUNCTION COMMENTED OUT FOR DEMO ONLY
# def process_pdf_with_streaming(pdf_file) -> None:
#     """Process PDF with streaming chat interface."""
#     # Check if we're already processing or have processed this file
#     if st.session_state.get("is_processing", False):
#         return
#     
#     chat_interface = st.session_state.get("chat_interface")
#     streaming_processor = st.session_state.get("streaming_processor")
#     
#     if not chat_interface or not streaming_processor:
#         st.error("Chat interface not properly initialized")
#         return
#     
#     # Start PDF processing
#     chat_interface.start_pdf_processing(pdf_file)
#     
#     # Create a placeholder for streaming content
#     message_placeholder = st.empty()
#     
#     # Process PDF with streaming
#     try:
#         full_response = ""
#         for chunk in streaming_processor.process_pdf_streaming(pdf_file, chat_interface):
#             full_response += chunk
#             # Add chunk to chat interface for real-time display
#             chat_interface.update_assistant_message(chunk, append=True)
#             # Update the message in real-time using placeholder
#             with message_placeholder.container():
#                 chat_interface.render_chat_container()
#             time.sleep(0.1)  # Small delay for smooth streaming
#         
#         
#         # Complete processing
#         extracted_data = chat_interface.get_extracted_data()
#         if extracted_data is not None:
#             # Convert to our internal format
#             df = convert_gemini_csv_to_internal_format(extracted_data)
#             
#             # Apply date range filter
#             start_date, end_date = get_current_date_range()
#             if start_date and end_date:
#                 df = filter_dataframe_by_date_range(df, start_date, end_date)
#             
#             st.session_state["df"] = df
#             st.session_state["analysis"] = analyze_dataframe(df)
#             
#             # Set AI summary
#             st.session_state["ai_pdf_summary"] = full_response
#         
#         chat_interface.complete_processing(extracted_data)
#         
#     except Exception as e:
#         st.error(f"Streaming processing failed: {str(e)}")
#         chat_interface.add_message("system", f"❌ Processing failed: {str(e)}")
#     finally:
#         # Always clear processing guard at end
#         st.session_state["is_processing"] = False


def process_demo_data_with_ai() -> None:
    """Process demo data with AI analysis using streaming chat interface."""
    # Check if demo data has already been processed
    if st.session_state.get("demo_ai_processed", False):
        return
    
    chat_interface = st.session_state.get("chat_interface")
    streaming_processor = st.session_state.get("streaming_processor")
    
    if not chat_interface or not streaming_processor:
        st.error("Chat interface not properly initialized")
        return
    
    # Load demo data
    df = load_demo_dataframe()
    
    # Start AI analysis for demo data
    chat_interface.start_demo_analysis()
    
    # Create a placeholder for streaming content
    message_placeholder = st.empty()
    
    # Process demo data with AI analysis
    try:
        full_response = ""
        for chunk in streaming_processor.process_demo_data_streaming(df, chat_interface):
            full_response += chunk
            # Add chunk to chat interface for real-time display
            chat_interface.update_assistant_message(chunk, append=True)
            # Update the message in real-time using placeholder
            with message_placeholder.container():
                chat_interface.render_chat_container()
            time.sleep(0.1)  # Small delay for smooth streaming
        
        
        # Apply date range filter
        start_date, end_date = get_current_date_range()
        if start_date and end_date:
            df = filter_dataframe_by_date_range(df, start_date, end_date)
        
        st.session_state["df"] = df
        st.session_state["analysis"] = analyze_dataframe(df)
        
        # Set AI summary
        st.session_state["ai_pdf_summary"] = full_response
        
        # Mark demo data as processed
        st.session_state["demo_ai_processed"] = True
        
        chat_interface.complete_processing(df)
        
    except Exception as e:
        st.error(f"Demo data AI analysis failed: {str(e)}")
        chat_interface.add_message("system", f"❌ Analysis failed: {str(e)}")
    finally:
        # Always clear processing guard at end
        st.session_state["is_processing"] = False


# PDF CONVERSION FUNCTION COMMENTED OUT FOR DEMO ONLY
# def convert_gemini_csv_to_internal_format(gemini_df) -> pd.DataFrame:
#     """Convert Gemini CSV output to internal DataFrame format."""
#     df = gemini_df.copy()
#     
#     # Rename columns to internal format
#     df = df.rename(columns={
#         "Transaction_Date": "timestamp",
#         "Amount": "amount", 
#         "Merchant_Category": "category",
#         "Balance_After": "balance_after",
#         "Description": "description",
#     })
#     
#     # Convert timestamp and ensure numeric types
#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
#     df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
#     
#     # Add required columns
#     df["currency"] = DEFAULT_CURRENCY
#     df["merchant"] = df["description"].astype(str)
#     df["category"] = df["category"].astype(str)
#     
#     # Clean up and validate
#     df = df.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)
#     
#     return df


def maybe_load_processed_data() -> None:
    """Load data if it has already been processed."""
    # Only load data if we have it and it's not currently being processed
    if (st.session_state.get("df") is not None and 
        st.session_state.get("processing_state") not in ["streaming", "uploading"]):
        
        # Check if we need to re-apply date filtering
        start_date, end_date = get_current_date_range()
        if start_date and end_date:
            df = st.session_state.get("df")
            if df is not None:
                # Only re-filter if the date range has actually changed
                current_start = st.session_state.get("last_filtered_start_date")
                current_end = st.session_state.get("last_filtered_end_date")
                
                if (current_start != start_date or current_end != end_date):
                    filtered_df = filter_dataframe_by_date_range(df, start_date, end_date)
                    st.session_state["df"] = filtered_df
                    st.session_state["analysis"] = analyze_dataframe(filtered_df)
                    # Remember the dates we filtered with
                    st.session_state["last_filtered_start_date"] = start_date
                    st.session_state["last_filtered_end_date"] = end_date


def main() -> None:
    """Application entry point."""
    set_page_config()

    # Init
    config = load_config()
    logger = get_logger()
    init_session_state()
    init_chat_components()

    render_header()
    render_sidebar()

    # Data source selection section
    data_source_area = st.container()
    with data_source_area:
        render_data_source_section()

    # Data layer - load data if already processed
    maybe_load_processed_data()

    # Content areas in responsive layout
    # Chat section for analysis
    chat_area = st.container()
    with chat_area:
        render_chat_section()

    viz_area = st.container()
    with viz_area:
        render_visualizations_section()

    # AI insights are now part of the chat flow, not a separate section

    # Empty state prompt
    render_empty_state()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
