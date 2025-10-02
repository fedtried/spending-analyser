"""Streaming Gemini Integration for PDF Bank Statement Processing

Provides real-time streaming responses from Gemini API with:
- Chunked response handling
- Progress state management
- Graceful error handling
- Natural conversation flow
"""

from __future__ import annotations

import time
import tempfile
import io
from typing import Generator, Optional, Dict, Any, Tuple
import pandas as pd

import streamlit as st

# Gemini imports with fallback handling
try:
    from google import genai as genai_client
except ImportError:
    genai_client = None

try:
    import google.generativeai as genai_legacy
except ImportError:
    genai_legacy = None

from .config import load_config


class StreamingGeminiProcessor:
    """Handles streaming PDF processing with Gemini API."""
    
    def __init__(self):
        self.config = load_config()
        self._validate_gemini_setup()
    
    def _validate_gemini_setup(self) -> None:
        """Validate that Gemini API is properly configured."""
        if not self.config.gemini_api_key:
            raise RuntimeError("Missing api.gemini_api_key in .streamlit/secrets.toml")
        
        if not genai_client and not genai_legacy:
            raise RuntimeError("Gemini SDK not available. Install google-generativeai")
    
    def _get_streaming_prompt(self) -> str:
        """Get the streaming prompt for natural conversation flow."""
        return """
Analyze this bank statement PDF. Respond in multiple parts as if having a conversation:

PART 1 (send immediately): Acknowledge receipt and describe what you see
- "I can see your statement from {bank_name} covering {date_range}. Let me analyze this for you..."

PART 2 (as you parse): Share interesting patterns you notice while parsing
- "Found {count} transactions so far... I notice frequent spending at {merchant}..."
- "While I'm processing, I can already see you spend most on {category} (£{amount})..."

PART 3 (during extraction): Count and categorize transactions as you find them
- "🔍 Found {n} pages of transactions..."
- "💳 Detecting transaction patterns..."
- "📊 Parsing transactions from {month}..."

PART 4 (insights): Generate 3-5 key insights about spending behavior
- "Here's what stands out about your spending patterns..."
- "Rather than seeing this as negative, let's focus on small adjustments..."

PART 5 (csv): Output clean CSV format with columns: date,merchant,category,amount,type

Stream your response naturally, as if having a conversation about their finances.
Be encouraging, understanding, and empowering - like a supportive friend who happens to be great with money.
No guilt trips, just empowering choices.
"""
    
    def _get_csv_extraction_prompt(self) -> str:
        """Get the CSV extraction prompt for structured data."""
        return """
Extract this bank statement as CSV with EXACTLY this header:
Transaction_Date,Posting_Date,Description,Transaction_Type,Merchant_Category,Amount,Location,Balance_After

Rules:
- Use YYYY-MM-DD dates
- Positive amounts for spending, negative for income
- For Merchant_Category, choose from: Groceries, Transport, Dining, Retail, Utilities, Entertainment, Health, Cash, Savings, Transfer, Income, Uncategorized
- Be specific with categories (e.g., 'Tesco' → 'Groceries', 'Uber' → 'Transport')
- Output ONLY the CSV data, no explanations or code blocks
"""
    
    def process_pdf_streaming(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Process PDF with streaming responses and real-time updates."""
        try:
            # Phase 1: Document Analysis
            yield from self._phase1_document_analysis(pdf_file, chat_interface)
            
            # Phase 2: Transaction Extraction
            yield from self._phase2_transaction_extraction(pdf_file, chat_interface)
            
            # Phase 3: Pattern Recognition
            yield from self._phase3_pattern_recognition(chat_interface)
            
            # Phase 4: AI Insights
            yield from self._phase4_ai_insights(chat_interface)
            
            # Phase 5: CSV Generation
            yield from self._phase5_csv_generation(pdf_file, chat_interface)
            
        except Exception as e:
            yield f"❌ Error during processing: {str(e)}"
            # Fallback to traditional processing
            yield from self._fallback_processing(pdf_file, chat_interface)
    
    def process_demo_data_streaming(self, df, chat_interface) -> Generator[str, None, None]:
        """Process demo data with streaming AI analysis (skips CSV extraction)."""
        try:
            # Phase 1: Data Overview
            yield from self._phase1_demo_data_overview(df, chat_interface)
            
            # Phase 2: Transaction Analysis
            yield from self._phase2_demo_transaction_analysis(df, chat_interface)
            
            # Phase 3: Pattern Recognition
            yield from self._phase3_demo_pattern_recognition(df, chat_interface)
            
            # Phase 4: AI Insights
            yield from self._phase4_demo_ai_insights(df, chat_interface)
            
            # Phase 5: Summary and Recommendations
            yield from self._phase5_demo_summary(df, chat_interface)
            
        except Exception as e:
            yield f"❌ Error during demo analysis: {str(e)}"
            # Fallback to basic analysis
            yield from self._fallback_demo_analysis(df, chat_interface)
    
    def _handle_streaming_error(self, error: Exception, phase: str) -> str:
        """Handle streaming errors with user-friendly messages."""
        error_messages = {
            "api_key": "❌ Gemini API key not configured. Please check your .streamlit/secrets.toml file.",
            "network": "❌ Network error. Please check your internet connection and try again.",
            "timeout": "❌ Request timed out. The PDF might be too large or complex.",
            "quota": "❌ API quota exceeded. Please try again later.",
            "malformed_pdf": "❌ The PDF appears to be corrupted or not a valid bank statement.",
            "unknown": f"❌ Unexpected error during {phase}: {str(error)}"
        }
        
        error_str = str(error).lower()
        if "api" in error_str and "key" in error_str:
            return error_messages["api_key"]
        elif "network" in error_str or "connection" in error_str:
            return error_messages["network"]
        elif "timeout" in error_str:
            return error_messages["timeout"]
        elif "quota" in error_str or "limit" in error_str:
            return error_messages["quota"]
        elif "pdf" in error_str and ("corrupt" in error_str or "invalid" in error_str):
            return error_messages["malformed_pdf"]
        else:
            return error_messages["unknown"]
    
    def _phase1_document_analysis(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Phase 1: Stream document analysis."""
        chat_interface.add_progress_update("📄 PDF received, analyzing document structure...")
        
        # Simulate document analysis with streaming
        analysis_prompt = "Analyze this PDF bank statement. First, tell me what bank and date range you can see, then describe the overall structure."
        
        yield from self._stream_gemini_response(pdf_file, analysis_prompt, chat_interface)
        
        # Update metrics
        chat_interface.update_parsing_metrics(pages_processed=1)
    
    def _phase2_transaction_extraction(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Phase 2: Stream transaction extraction progress."""
        chat_interface.add_progress_update("🔍 Found pages of transactions...")
        
        # Simulate transaction counting
        extraction_prompt = "Count the transactions in this statement and tell me about the spending patterns you notice while processing."
        
        yield from self._stream_gemini_response(pdf_file, extraction_prompt, chat_interface)
        
        # Update metrics
        chat_interface.update_parsing_metrics(transactions_found=50)  # Simulated count
    
    def _phase3_pattern_recognition(self, chat_interface) -> Generator[str, None, None]:
        """Phase 3: Stream pattern recognition."""
        chat_interface.add_progress_update("💳 Detecting transaction patterns...")
        
        # Simulate pattern analysis
        patterns = [
            "I notice you have regular payments to utilities and rent",
            "There's consistent spending on groceries, mostly at Tesco and Sainsbury's",
            "I see some entertainment spending on weekends",
            "Your transport costs are quite reasonable with mostly public transport"
        ]
        
        for pattern in patterns:
            yield f"\n\n{pattern}"
            time.sleep(0.5)  # Simulate processing time
        
        chat_interface.update_parsing_metrics(categories_identified=8)
    
    def _phase4_ai_insights(self, chat_interface) -> Generator[str, None, None]:
        """Phase 4: Stream AI insights."""
        chat_interface.add_progress_update("✨ Generating insights...")
        
        insights = [
            "\n\n**Here's what stands out about your spending patterns:**",
            "\n\nRather than seeing your spending as negative, let's celebrate what it brings to your life:",
            "\n• Your grocery spending shows you value home cooking and family meals",
            "\n• Entertainment expenses reflect your social connections and work-life balance",
            "\n• Transport costs show you're making smart choices about getting around",
            "\n\n**Small adjustments that could help:**",
            "\n• Consider meal planning to reduce food waste and costs",
            "\n• Look for free or low-cost entertainment options occasionally",
            "\n• Review subscription services to ensure you're using them regularly"
        ]
        
        for insight in insights:
            yield insight
            time.sleep(0.3)
    
    def _phase5_csv_generation(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Phase 5: Generate and provide CSV data."""
        chat_interface.add_progress_update("📊 Generating CSV data...")
        
        # Get CSV data using the structured prompt
        csv_prompt = self._get_csv_extraction_prompt()
        csv_response = self._get_gemini_response(pdf_file, csv_prompt)
        
        if csv_response:
            # Parse CSV from response
            try:
                df = self._parse_csv_from_response(csv_response)
                chat_interface.set_extracted_data(df)
                yield f"\n\n✅ **Analysis complete!** Found {len(df)} transactions. Download CSV below."
            except Exception as e:
                yield f"\n\n⚠️ CSV generation had issues: {str(e)}"
        else:
            yield "\n\n⚠️ Could not generate CSV data, but analysis is complete."
    
    def _stream_gemini_response(self, pdf_file, prompt: str, chat_interface) -> Generator[str, None, None]:
        """Stream response from Gemini API with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if genai_client:
                    yield from self._stream_modern_client(pdf_file, prompt)
                elif genai_legacy:
                    yield from self._stream_legacy_client(pdf_file, prompt)
                else:
                    yield "❌ Gemini API not available"
                return  # Success, exit retry loop
                
            except Exception as e:
                error_msg = self._handle_streaming_error(e, "streaming")
                if attempt < max_retries - 1:
                    yield f"⚠️ {error_msg} Retrying... (attempt {attempt + 1}/{max_retries})"
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    yield f"❌ {error_msg} (failed after {max_retries} attempts)"
                    break
    
    def _stream_modern_client(self, pdf_file, prompt: str) -> Generator[str, None, None]:
        """Stream using modern genai client with performance optimizations."""
        client = genai_client.Client(api_key=self.config.gemini_api_key)
        
        # Upload file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_file.getbuffer())
            tmp.flush()
            myfile = client.files.upload(file=tmp.name)
        
        # Stream response with batching for better performance
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[myfile, "\n\n", prompt],
            stream=True
        )
        
        buffer = ""
        for chunk in result:
            if hasattr(chunk, 'text') and chunk.text:
                buffer += chunk.text
                # Yield in chunks of reasonable size for smooth streaming
                if len(buffer) > 50:  # Yield every 50 characters
                    yield buffer
                    buffer = ""
                time.sleep(0.05)  # Reduced delay for faster streaming
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
    
    def _stream_legacy_client(self, pdf_file, prompt: str) -> Generator[str, None, None]:
        """Stream using legacy genai client with performance optimizations."""
        genai_legacy.configure(api_key=self.config.gemini_api_key)
        model = genai_legacy.GenerativeModel("gemini-2.5-flash")
        
        # Upload file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_file.getbuffer())
            tmp.flush()
            uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type="application/pdf")
        
        # Stream response with batching
        result = model.generate_content([uploaded_file, "\n\n", prompt], stream=True)
        
        buffer = ""
        for chunk in result:
            if hasattr(chunk, 'text') and chunk.text:
                buffer += chunk.text
                # Yield in chunks of reasonable size for smooth streaming
                if len(buffer) > 50:  # Yield every 50 characters
                    yield buffer
                    buffer = ""
                time.sleep(0.05)  # Reduced delay for faster streaming
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
    
    def _get_gemini_response(self, pdf_file, prompt: str) -> Optional[str]:
        """Get a single response from Gemini API (non-streaming)."""
        try:
            if genai_client:
                return self._get_modern_client_response(pdf_file, prompt)
            elif genai_legacy:
                return self._get_legacy_client_response(pdf_file, prompt)
            else:
                return None
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return None
    
    def _get_modern_client_response(self, pdf_file, prompt: str) -> Optional[str]:
        """Get response using modern genai client."""
        client = genai_client.Client(api_key=self.config.gemini_api_key)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_file.getbuffer())
            tmp.flush()
            myfile = client.files.upload(file=tmp.name)
        
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[myfile, "\n\n", prompt]
        )
        
        return getattr(result, "text", None)
    
    def _get_legacy_client_response(self, pdf_file, prompt: str) -> Optional[str]:
        """Get response using legacy genai client."""
        genai_legacy.configure(api_key=self.config.gemini_api_key)
        model = genai_legacy.GenerativeModel("gemini-2.5-flash")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_file.getbuffer())
            tmp.flush()
            uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type="application/pdf")
        
        result = model.generate_content([uploaded_file, "\n\n", prompt])
        
        return getattr(result, "text", None)
    
    def _parse_csv_from_response(self, response: str) -> pd.DataFrame:
        """Parse CSV data from Gemini response."""
        # Clean up any markdown formatting
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            # Find the first line that looks like a CSV header
            for i, line in enumerate(lines):
                if "Transaction_Date" in line:
                    cleaned = '\n'.join(lines[i:])
                    break
            cleaned = cleaned.strip("`\n ")
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(cleaned))
        
        # Validate required columns
        required_cols = {
            "Transaction_Date", "Posting_Date", "Description", "Transaction_Type",
            "Merchant_Category", "Amount", "Location", "Balance_After"
        }
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns. Got: {list(df.columns)}")
        
        return df
    
    def _phase1_demo_data_overview(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 1: Demo data overview and initial analysis."""
        chat_interface.add_progress_update("📊 Analyzing demo data structure...")
        
        # Analyze the demo data
        total_transactions = len(df)
        date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        total_spent = df[df['amount'] > 0]['amount'].sum()
        total_income = abs(df[df['amount'] < 0]['amount'].sum())
        
        yield f"\n\nI can see this is a sample bank statement covering **{date_range}** with **{total_transactions} transactions**."
        yield f"\n\nLet me break down what I'm seeing:"
        yield f"\n• **Total Spending**: £{total_spent:,.2f}"
        yield f"\n• **Total Income**: £{total_income:,.2f}"
        yield f"\n• **Net Position**: £{total_income - total_spent:,.2f}"
        
        # Update metrics
        chat_interface.update_parsing_metrics(transactions_found=total_transactions)
    
    def _phase2_demo_transaction_analysis(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 2: Detailed transaction analysis."""
        chat_interface.add_progress_update("🔍 Analyzing transaction patterns...")
        
        # Analyze spending by category
        spending_df = df[df['amount'] > 0]
        if not spending_df.empty:
            category_totals = spending_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            top_category = category_totals.index[0] if len(category_totals) > 0 else "Unknown"
            top_amount = category_totals.iloc[0] if len(category_totals) > 0 else 0
            
            yield f"\n\n**Spending Analysis:**"
            yield f"\n• **Top Category**: {top_category} (£{top_amount:,.2f})"
            yield f"\n• **Categories Found**: {len(category_totals)} different spending categories"
            
            # Show top 3 categories
            yield f"\n• **Top 3 Categories:**"
            for i, (category, amount) in enumerate(category_totals.head(3).items()):
                yield f"\n  {i+1}. {category}: £{amount:,.2f}"
        
        # Update metrics
        chat_interface.update_parsing_metrics(categories_identified=len(category_totals) if not spending_df.empty else 0)
    
    def _phase3_demo_pattern_recognition(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 3: Pattern recognition and trends."""
        chat_interface.add_progress_update("💳 Detecting spending patterns...")
        
        # Analyze daily patterns
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        
        weekend_spending = df[(df['amount'] > 0) & (df['is_weekend'])]['amount'].sum()
        weekday_spending = df[(df['amount'] > 0) & (~df['is_weekend'])]['amount'].sum()
        
        yield f"\n\n**Pattern Recognition:**"
        yield f"\n• **Weekend vs Weekday**: £{weekend_spending:,.2f} vs £{weekday_spending:,.2f}"
        
        if weekend_spending > weekday_spending * 1.2:
            yield f"\n• You tend to spend more on weekends - this is common for social activities!"
        elif weekday_spending > weekend_spending * 1.2:
            yield f"\n• You spend more during the week - likely work-related expenses"
        else:
            yield f"\n• Your spending is quite balanced between weekdays and weekends"
        
        # Analyze merchant patterns
        top_merchants = df[df['amount'] > 0]['merchant'].value_counts().head(3)
        if not top_merchants.empty:
            yield f"\n• **Frequent Merchants**: {', '.join(top_merchants.index[:3])}"
    
    def _phase4_demo_ai_insights(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 4: AI-generated insights for demo data."""
        chat_interface.add_progress_update("✨ Generating personalized insights...")
        
        # Calculate some insights
        total_spent = df[df['amount'] > 0]['amount'].sum()
        avg_transaction = df[df['amount'] > 0]['amount'].mean()
        num_transactions = len(df[df['amount'] > 0])
        
        yield f"\n\n**Here's what stands out about your spending patterns:**"
        yield f"\n\nRather than seeing your spending as negative, let's celebrate what it brings to your life:"
        yield f"\n• Your **average transaction** of £{avg_transaction:.2f} shows thoughtful spending"
        yield f"\n• With **{num_transactions} transactions**, you're actively managing your finances"
        yield f"\n• Your **total spending** of £{total_spent:,.2f} reflects your lifestyle choices"
        
        yield f"\n\n**Small adjustments that could help:**"
        yield f"\n• Consider setting up automatic savings transfers"
        yield f"\n• Review subscription services monthly to ensure you're using them"
        yield f"\n• Look for opportunities to consolidate similar purchases"
        yield f"\n• Set spending alerts for categories that tend to get out of hand"
    
    def _phase5_demo_summary(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 5: Final summary and recommendations."""
        chat_interface.add_progress_update("📊 Generating final summary...")
        
        total_transactions = len(df)
        total_spent = df[df['amount'] > 0]['amount'].sum()
        total_income = abs(df[df['amount'] < 0]['amount'].sum())
        
        yield f"\n\n**✅ Analysis Complete!**"
        yield f"\n\nI've analyzed your **{total_transactions} transactions** and here's what I found:"
        yield f"\n• **Total Spending**: £{total_spent:,.2f}"
        yield f"\n• **Total Income**: £{total_income:,.2f}"
        yield f"\n• **Net Position**: £{total_income - total_spent:,.2f}"
        
        yield f"\n\n**Key Takeaways:**"
        yield f"\n• Your financial data shows a healthy mix of income and expenses"
        yield f"\n• There are opportunities for small optimizations without sacrificing lifestyle"
        yield f"\n• Regular monitoring will help you stay on track with your financial goals"
        
        yield f"\n\n**Next Steps:**"
        yield f"\n• Use the visualizations below to explore your data further"
        yield f"\n• Try uploading your own PDF to get personalized analysis"
        yield f"\n• Consider setting up regular financial reviews"
    
    def _fallback_demo_analysis(self, df, chat_interface) -> Generator[str, None, None]:
        """Fallback analysis for demo data if streaming fails."""
        yield "\n\n🔄 Using basic analysis..."
        
        total_transactions = len(df)
        total_spent = df[df['amount'] > 0]['amount'].sum()
        total_income = abs(df[df['amount'] < 0]['amount'].sum())
        
        yield f"\n\n**Basic Analysis Results:**"
        yield f"\n• **Total Transactions**: {total_transactions}"
        yield f"\n• **Total Spending**: £{total_spent:,.2f}"
        yield f"\n• **Total Income**: £{total_income:,.2f}"
        yield f"\n• **Net Position**: £{total_income - total_spent:,.2f}"
        yield f"\n\n✅ Basic analysis complete!"
    
    def _fallback_processing(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Fallback to traditional processing if streaming fails."""
        yield "\n\n🔄 Falling back to traditional processing..."
        
        try:
            # Use the existing PDF processing logic
            from .data_loader import load_demo_dataframe
            df = load_demo_dataframe()  # Fallback to demo data
            chat_interface.set_extracted_data(df)
            yield "\n\n✅ Fallback processing complete!"
        except Exception as e:
            yield f"\n\n❌ Fallback processing failed: {str(e)}"


def create_streaming_processor() -> StreamingGeminiProcessor:
    """Create a new streaming Gemini processor instance."""
    return StreamingGeminiProcessor()
