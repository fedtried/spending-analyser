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
        "data_source": "Demo Data",  # or "Upload PDF"
        "analysis_period": "Custom Date Range",
        "date_range_type": "Custom Date Range",  # "Last 7 days", "Last 30 days", etc. or "Custom Date Range"
        "start_date": date(2025, 7, 1),  # Default to demo data range start
        "end_date": date(2025, 9, 30),   # Default to demo data range end
        "currency": DEFAULT_CURRENCY,
        "df": None,  # placeholder for DataFrame when added later
        "analysis": None,  # AnalysisResult placeholder
        "ai_pdf_summary": None,  # Stores one-sentence summary 
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

        st.caption("Use the controls to configure your analysis.")

        st.markdown("**Data Source**")
        data_source = st.radio(
            label="Choose data source",
            options=["Demo Data", "Upload PDF"],
            index=0 if st.session_state["data_source"] == "Demo Data" else 1,
            horizontal=True,
            key="data_source",
            help="Load prebuilt demo data or upload your own PDF bank statement.",
        )

        if st.session_state.get("data_source") == "Upload PDF":
            uploaded = st.file_uploader(
                "Upload PDF bank statement",
                type=["pdf"],
                accept_multiple_files=False,
                help="Upload your bank statement PDF. The app will extract transaction data automatically.",
            )
            st.session_state["uploaded_file"] = uploaded

        st.markdown("**Analysis Period**")
        date_range_type = st.selectbox(
            label="Time range type",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date", "Custom Date Range"],
            index=["Last 7 days", "Last 30 days", "Last 90 days", "Year to date", "Custom Date Range"].index(
                st.session_state.get("date_range_type", "Custom Date Range")
            ),
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
                    value=st.session_state.get("start_date") or date(2025, 7, 1),
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=st.session_state.get("end_date") or date(2025, 9, 30),
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


def render_summary_section() -> None:
    """Render the summary statistics section, using demo data if available."""
    with st.container(border=True):
        st.markdown("### Overview 📊")
        
        # Show current date range
        start_date, end_date = get_current_date_range()
        if start_date and end_date:
            st.caption(f"Analysis period: {start_date} to {end_date}")

        analysis = st.session_state.get("analysis")
        if analysis is None:
            st.write(
                "Upload your data or load demo data to see personalized summaries, trends, and spending breakdowns."
            )
            render_metrics_row()
        else:
            render_metrics_row(
                total_spent=analysis.total_spent,
                total_income=analysis.total_income,
                total_net=analysis.total_net,
            )



def render_visualizations_section() -> None:
    """Render the visualizations placeholder section with a minimal chart."""
    with st.container(border=True):
        st.markdown("### Visualizations 🎯")
        try:
            df = st.session_state.get("df")

            if df is None or df.empty:
                st.info("Visualization will appear here once data is available.")
                return

            # Daily Income vs Spending with Balance Trend Overlay
            df_bal = df.copy()
            df_bal["Day"] = df_bal["timestamp"].dt.date
            
            # Calculate daily income and spending based on amount sign
            # In demo data: negative amounts = income, positive amounts = spending
            daily_income = df_bal[df_bal["amount"] < 0].groupby("Day")["amount"].sum().reset_index(name="Income")
            daily_income["Income"] = -daily_income["Income"]  # Convert to positive for display
            
            daily_spending = df_bal[df_bal["amount"] > 0].groupby("Day")["amount"].sum().reset_index(name="Spending")
            
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
            
            spend_df = df[df["amount"] > 0].copy()
            if not spend_df.empty:
                spend_df["Spend"] = spend_df["amount"]
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


def render_ai_insights_section() -> None:
    """Render the AI insights placeholder section."""
    with st.container(border=True):
        st.markdown("### AI Insights 🤖")
        summary = st.session_state.get("ai_pdf_summary")
        if summary:
            st.write(summary)
        else:
            st.info("Upload a PDF to get AI insights.")


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
        return st.session_state.get("start_date"), st.session_state.get("end_date")
    else:
        # For preset ranges, get from session state
        return st.session_state.get("start_date"), st.session_state.get("end_date")


def maybe_load_demo_data() -> None:
    """Populate session with demo data and analysis if Demo Data is selected."""
    if st.session_state.get("data_source") == "Demo Data":
        df = load_demo_dataframe()
        
        # Apply date range filter
        start_date, end_date = get_current_date_range()
        if start_date and end_date:
            df = filter_dataframe_by_date_range(df, start_date, end_date)
        
        st.session_state["df"] = df
        st.session_state["analysis"] = analyze_dataframe(df)
    elif st.session_state.get("data_source") == "Upload PDF":
        file_obj = st.session_state.get("uploaded_file")
        if file_obj is not None:
            try:
                gemini_csv_df: Optional[pd.DataFrame] = None
                try:
                    cfg = load_config()
                    if not cfg.gemini_api_key:
                        raise RuntimeError("Missing api.gemini_api_key in .streamlit/secrets.toml")

                    instruction = (
                        "Extract this bank statement as CSV with EXACTLY this header: "
                        "Transaction_Date,Posting_Date,Description,Transaction_Type,Merchant_Category,Amount,Location,Balance_After\n"
                        "Rules: Use YYYY-MM-DD dates. Positive amounts for spending, negative for income. "
                        "For Merchant_Category, choose from: Groceries, Transport, Dining, Retail, Utilities, Entertainment, Health, Cash, Savings, Transfer, Income, Uncategorized. "
                        "Be specific with categories (e.g., 'Tesco' → 'Groceries', 'Uber' → 'Transport'). "
                        "Output ONLY the CSV data, no explanations or code blocks."
                    )

                    csv_text: Optional[str] = None

                    # Modern client flow
                    if genai_client is not None:
                        import tempfile
                        client = genai_client.Client(api_key=cfg.gemini_api_key)
                        mime_type = getattr(file_obj, "type", "application/pdf") or "application/pdf"
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(file_obj.getbuffer())
                            tmp.flush()
                            myfile = client.files.upload(file=tmp.name)
                        result = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[myfile, "\n\n", instruction],
                        )
                        csv_text = getattr(result, "text", None) or None

                    # Legacy client flow
                    elif genai_legacy is not None:
                        import tempfile
                        genai_legacy.configure(api_key=cfg.gemini_api_key)
                        model = genai_legacy.GenerativeModel("gemini-2.5-flash")
                        mime_type = getattr(file_obj, "type", "application/pdf") or "application/pdf"
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(file_obj.getbuffer())
                            tmp.flush()
                            try:
                                uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type=mime_type)
                            except TypeError:
                                uploaded_file = genai_legacy.upload_file(file=tmp.name, mime_type=mime_type)
                        result = model.generate_content([uploaded_file, "\n\n", instruction])
                        csv_text = getattr(result, "text", None)
                        if not csv_text and hasattr(result, "candidates"):
                            try:
                                csv_text = result.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                            except Exception:
                                csv_text = None
                    else:
                        raise RuntimeError("Gemini SDK not available. Did you install google-generativeai?")

                    if csv_text:
                        # Clean up any markdown formatting
                        cleaned = csv_text.strip()
                        if cleaned.startswith("```"):
                            lines = cleaned.split('\n')
                            # Find the first line that looks like a CSV header
                            for i, line in enumerate(lines):
                                if "Transaction_Date" in line:
                                    cleaned = '\n'.join(lines[i:])
                                    break
                            cleaned = cleaned.strip("`\n ")
                        
                        # Parse CSV
                        gemini_csv_df = pd.read_csv(io.StringIO(cleaned))
                        
                        # Validate required columns
                        required_cols = {
                            "Transaction_Date","Posting_Date","Description","Transaction_Type",
                            "Merchant_Category","Amount","Location","Balance_After"
                        }
                        if not required_cols.issubset(set(gemini_csv_df.columns)):
                            raise ValueError(f"CSV missing required columns. Got: {list(gemini_csv_df.columns)}")
                except Exception as gem_csv_err:
                    st.caption(f"Gemini CSV note: {gem_csv_err}")

                # Prefer Gemini CSV if available; else fallback to local parser
                if gemini_csv_df is not None and not gemini_csv_df.empty:
                    # Use Gemini's structured output directly
                    df = gemini_csv_df.rename(columns={
                        "Transaction_Date": "timestamp",
                        "Amount": "amount", 
                        "Merchant_Category": "category",
                        "Balance_After": "balance_after",
                        "Description": "description",
                    })
                    
                    # Convert timestamp and ensure numeric types
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
                    
                    # Add required columns
                    df["currency"] = DEFAULT_CURRENCY
                    df["merchant"] = df["description"].astype(str)
                    df["category"] = df["category"].astype(str)
                    
                    # Clean up and validate
                    df = df.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)
                else:
                    df = load_user_dataframe(file_obj)

                # Apply date range filter
                start_date, end_date = get_current_date_range()
                if start_date and end_date:
                    df = filter_dataframe_by_date_range(df, start_date, end_date)
                
                st.session_state["df"] = df
                st.session_state["analysis"] = analyze_dataframe(df)

                # Gemini integration: upload PDF and get personalized summary
                try:
                    cfg = load_config()
                    if not cfg.gemini_api_key:
                        raise RuntimeError("Missing api.gemini_api_key in .streamlit/secrets.toml")

                    summary_text = None

                    # Preferred modern client flow: google.genai Client + Files API
                    if genai_client is not None:
                        import tempfile
                        client = genai_client.Client(api_key=cfg.gemini_api_key)

                        # Persist uploaded file to disk for upload API
                        mime_type = getattr(file_obj, "type", "application/pdf") or "application/pdf"
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(file_obj.getbuffer())
                            tmp.flush()
                            myfile = client.files.upload(file=tmp.name)  
                        result = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                myfile,
                                "\n\n",
                                "Write a personalized, summary of this bank statement. Focus on what the spending brings to the person's life, and offer suggestions for maintaining balance. Be encouraging and understanding, like a supportive friend who happens to be great with money. Keep it to 2-3 sentences.",
                            ],
                        )
                        summary_text = getattr(result, "text", None) or ""

                    # Legacy fallback: google.generativeai with file upload
                    elif genai_legacy is not None:
                        import tempfile
                        genai_legacy.configure(api_key=cfg.gemini_api_key)
                        model = genai_legacy.GenerativeModel("gemini-2.5-flash")

                        # Persist uploaded file then upload via legacy Files helper
                        mime_type = getattr(file_obj, "type", "application/pdf") or "application/pdf"
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(file_obj.getbuffer())
                            tmp.flush()
                            try:
                                uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type=mime_type)
                            except TypeError:
                                # Some versions expect 'file' kwarg
                                uploaded_file = genai_legacy.upload_file(file=tmp.name, mime_type=mime_type)

                        result = model.generate_content([
                            uploaded_file,
                            "\n\n",
                            "Write a personalized, summary of this bank statement. Focus on what the spending brings to the person's life, and offer suggestions for maintaining balance. Be encouraging and understanding, like a supportive friend who happens to be great with money. Keep it to 2-3 sentences.",
                        ])

                        # Try to extract text depending on SDK version
                        summary_text = getattr(result, "text", None)
                        if not summary_text and hasattr(result, "candidates"):
                            try:
                                summary_text = result.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                            except Exception:
                                summary_text = None
                    else:
                        raise RuntimeError("Gemini SDK not available. Did you install google-generativeai?")

                    if summary_text:
                        st.session_state["ai_pdf_summary"] = summary_text.strip()
                    else:
                        st.session_state["ai_pdf_summary"] = ""
                except Exception as gem_err:
                    st.error(f"Gemini call failed: {gem_err}")
            except Exception as exc:
                st.session_state["df"] = None
                st.session_state["analysis"] = None
                st.error("Failed to process PDF. Ensure it's a valid bank statement PDF.")
                st.caption(f"Parser note: {exc}")


def main() -> None:
    """Application entry point."""
    set_page_config()

    # Init
    config = load_config()
    logger = get_logger()
    init_session_state()

    render_header()
    render_sidebar()

    # Data layer - reload data if date range has changed
    maybe_load_demo_data()


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
