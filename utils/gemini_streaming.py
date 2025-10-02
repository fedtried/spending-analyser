"""Streaming Gemini Integration 
"""

from __future__ import annotations

import tempfile
import io
import json
from typing import Generator, Optional, Dict, Any, Tuple
import pandas as pd

import streamlit as st

# Handle different Gemini SDK versions
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
    
    def process_demo_data_streaming(self, df, chat_interface) -> Generator[str, None, None]:
        """Process demo data with Gemini AI analysis."""
        try:
            yield from self._phase1_demo_data_overview(df, chat_interface)
            yield from self._phase2_demo_transaction_analysis(df, chat_interface)
            yield from self._phase3_real_ai_insights(df, chat_interface)
            yield from self._phase4_demo_summary(df, chat_interface)
            
        except Exception as e:
            yield f"⌛ Error during demo analysis: {str(e)}"
            yield from self._fallback_demo_analysis(df, chat_interface)
    
    def _phase1_demo_data_overview(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 1: Demo data overview - factual analysis."""
        chat_interface.add_progress_update("📊 Analyzing demo data structure...")
        
        total_transactions = len(df)
        date_range = f"{df['timestamp'].min().strftime('%B %d')} to {df['timestamp'].max().strftime('%B %d, %Y')}"
        total_spent = abs(df[df['amount'] < 0]['amount'].sum())
        total_income = df[df['amount'] > 0]['amount'].sum()
        
        yield f"\n\nI can see this bank statement covers **{date_range}** "
        yield f"with **{total_transactions} transactions**.\n\n"
        yield "Let me analyze your spending patterns:\n"
        yield f"• **Total Spending**: £{total_spent:,.0f}\n"
        yield f"• **Total Income**: £{total_income:,.0f}\n"
        yield f"• **Net Position**: £{total_income - total_spent:+,.0f}"
        
        chat_interface.update_parsing_metrics(transactions_found=total_transactions)
    
    def _phase2_demo_transaction_analysis(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 2: Category breakdown - factual analysis."""
        chat_interface.add_progress_update("🔍 Analyzing transaction patterns...")
        
        spending_df = df[df['amount'] < 0]
        if not spending_df.empty:
            category_totals = (-spending_df['amount']).groupby(spending_df['category']).sum().sort_values(ascending=False)
            
            yield "\n\n**Spending by Category:**\n"
            
            for i, (category, amount) in enumerate(category_totals.head(5).items()):
                percentage = (amount / category_totals.sum()) * 100
                yield f"{i+1}. **{category}**: £{amount:.0f} ({percentage:.1f}%)\n"
        
        chat_interface.update_parsing_metrics(categories_identified=len(category_totals) if not spending_df.empty else 0)
    
    def _phase3_real_ai_insights(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 3: AI-generated insights using Gemini API."""
        chat_interface.add_progress_update("✨ Generating AI insights...")
        
        spending_df = df[df['amount'] < 0]
        category_totals = (-spending_df['amount']).groupby(spending_df['category']).sum().sort_values(ascending=False)
        
        data_summary = {
            "total_transactions": len(df),
            "total_spent": float(abs(spending_df['amount'].sum())),
            "total_income": float(df[df['amount'] > 0]['amount'].sum()),
            "date_range": f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
            "top_categories": {cat: float(amt) for cat, amt in category_totals.head(5).items()},
            "transaction_count_by_category": spending_df['category'].value_counts().to_dict(),
            "average_transaction": float(spending_df['amount'].mean()),
            "largest_expense": float(spending_df['amount'].min()),
            "weekend_vs_weekday": {
                "weekend_spend": float(abs(df[(df['amount'] < 0) & (df['timestamp'].dt.dayofweek >= 5)]['amount'].sum())),
                "weekday_spend": float(abs(df[(df['amount'] < 0) & (df['timestamp'].dt.dayofweek < 5)]['amount'].sum()))
            },
            "top_merchants": df[df['amount'] < 0]['merchant'].value_counts().head(5).to_dict()
        }
        
        insights_prompt = f"""
        Analyze this spending data and provide personalized, empathetic financial insights.
        
        Data Summary:
        {json.dumps(data_summary, indent=2)}
        
        Provide insights in a conversational, supportive tone that:
        1. Celebrates what their spending enables in their life (social connections, health, etc.)
        2. Identifies 2-3 specific patterns or unusual observations
        3. Suggests 2-3 actionable optimizations with specific pound amounts they could save
        4. Avoids being preachy or judgmental
        5. Speaks like a supportive friend who understands money
        
        Format your response in clear sections but keep it natural and conversational.
        Be specific with amounts and percentages. Make it feel personalized to THIS data.
        """
        
        try:
            yield "\n\n**AI Analysis of Your Spending:**\n\n"
            
            for chunk in self._stream_gemini_text_response(insights_prompt):
                yield chunk
                
        except Exception as e:
            yield f"\n\n⚠️ AI service temporarily unavailable. Here's a basic analysis:\n\n"
            yield from self._generate_basic_insights(data_summary)
    
    def _stream_gemini_text_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream response from Gemini API for text-only prompts."""
        try:
            if genai_client:
                client = genai_client.Client(api_key=self.config.gemini_api_key)
                result = client.models.generate_content(
                    model="gemini-2.0-flash-exp",  # Latest model for better insights
                    contents=prompt,
                    generation_config={
                        "temperature": 0.7,  # Some creativity for insights
                        "top_p": 0.95,
                        "max_output_tokens": 1000,
                    },
                    stream=True
                )
                
                buffer = ""
                for chunk in result:
                    if hasattr(chunk, 'text') and chunk.text:
                        buffer += chunk.text
                        # Stream by sentences
                        sentences = buffer.split('. ')
                        while len(sentences) > 1:
                            sentence = sentences.pop(0)
                            yield sentence + '. '
                        buffer = sentences[0] if sentences else ""
                
                if buffer:
                    yield buffer
                    
            elif genai_legacy:
                genai_legacy.configure(api_key=self.config.gemini_api_key)
                model = genai_legacy.GenerativeModel("gemini-2.0-flash-exp")
                
                result = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_output_tokens": 1000,
                    },
                    stream=True
                )
                
                buffer = ""
                for chunk in result:
                    if hasattr(chunk, 'text') and chunk.text:
                        buffer += chunk.text
                        # Stream by sentences
                        sentences = buffer.split('. ')
                        while len(sentences) > 1:
                            sentence = sentences.pop(0)
                            yield sentence + '. '
                        buffer = sentences[0] if sentences else ""
                
                if buffer:
                    yield buffer
                    
            else:
                raise Exception("Gemini API not available")
                
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _generate_basic_insights(self, data_summary: dict) -> Generator[str, None, None]:
        """Generate basic insights if Gemini fails."""
        top_cat = list(data_summary['top_categories'].keys())[0] if data_summary['top_categories'] else "Unknown"
        top_amount = list(data_summary['top_categories'].values())[0] if data_summary['top_categories'] else 0
        
        yield f"Your highest spending category is **{top_cat}** at £{top_amount:.0f}.\n"
        
        if data_summary['weekend_vs_weekday']['weekend_spend'] > data_summary['weekend_vs_weekday']['weekday_spend']:
            yield "You tend to spend more on weekends, which suggests an active social life.\n"
        else:
            yield "Your weekday spending exceeds weekend spending, likely due to work-related expenses.\n"
        
        yield f"\nWith total spending of £{data_summary['total_spent']:.0f} against income of £{data_summary['total_income']:.0f}, "
        yield f"you have a net position of £{data_summary['total_income'] - data_summary['total_spent']:+.0f}.\n"
    
    def _phase4_demo_summary(self, df, chat_interface) -> Generator[str, None, None]:
        """Phase 4: Final summary."""
        chat_interface.add_progress_update("📊 Completing analysis...")
        
        yield "\n\n**✅ Analysis Complete!**\n\n"
        yield "Use the visualizations below to explore your spending patterns in detail. "
        yield "Consider uploading your own bank statement for personalized insights tailored to your actual spending."
    
    def _fallback_demo_analysis(self, df, chat_interface) -> Generator[str, None, None]:
        """Fallback if all else fails."""
        yield "\n\n📄 Using basic analysis...\n"
        
        total_transactions = len(df)
        total_spent = abs(df[df['amount'] < 0]['amount'].sum())
        total_income = df[df['amount'] > 0]['amount'].sum()
        
        yield "\n**Basic Analysis Results:**\n"
        yield f"• **Total Transactions**: {total_transactions}\n"
        yield f"• **Total Spending**: £{total_spent:,.2f}\n"
        yield f"• **Total Income**: £{total_income:,.2f}\n"
        yield f"• **Net Position**: £{total_income - total_spent:+,.2f}\n"
        yield "\n✅ Analysis complete!"
    
    # PDF processing methods remain the same but should also use real Gemini calls
    def process_pdf_streaming(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Process PDF with Gemini API calls."""
        try:
            # Actually analyze the PDF with Gemini
            prompt = """
            Analyze this bank statement PDF and provide:
            1. Overview of the bank, date range, and number of transactions
            2. Spending breakdown by category with specific amounts
            3. Personalized insights about spending patterns
            4. 3 specific recommendations with pound amounts they could save
            
            Be conversational, empathetic, and specific. Focus on what their spending enables in their life.
            """
            
            chat_interface.add_progress_update("📄 Analyzing your bank statement with AI...")
            
            # Stream real Gemini response for PDF
            for chunk in self._stream_gemini_pdf_response(pdf_file, prompt, chat_interface):
                yield chunk
            
            # Then extract CSV data
            yield from self._extract_csv_from_pdf(pdf_file, chat_interface)
            
        except Exception as e:
            yield f"⌛ Error during PDF processing: {str(e)}"
            yield from self._fallback_processing(pdf_file, chat_interface)
    
    def _stream_gemini_pdf_response(self, pdf_file, prompt: str, chat_interface) -> Generator[str, None, None]:
        """Stream Gemini response for PDF analysis."""
        try:
            if genai_client:
                client = genai_client.Client(api_key=self.config.gemini_api_key)
                
                # Upload PDF
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(pdf_file.getbuffer())
                    tmp.flush()
                    myfile = client.files.upload(file=tmp.name)
                
                # Get streaming response
                result = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[myfile, "\n\n", prompt],
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 2000,
                    },
                    stream=True
                )
                
                buffer = ""
                for chunk in result:
                    if hasattr(chunk, 'text') and chunk.text:
                        buffer += chunk.text
                        sentences = buffer.split('. ')
                        while len(sentences) > 1:
                            sentence = sentences.pop(0)
                            yield sentence + '. '
                        buffer = sentences[0] if sentences else ""
                
                if buffer:
                    yield buffer
                    
            elif genai_legacy:
                # Similar for legacy client
                genai_legacy.configure(api_key=self.config.gemini_api_key)
                model = genai_legacy.GenerativeModel("gemini-2.0-flash-exp")
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(pdf_file.getbuffer())
                    tmp.flush()
                    uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type="application/pdf")
                
                result = model.generate_content(
                    [uploaded_file, "\n\n", prompt],
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 2000,
                    },
                    stream=True
                )
                
                buffer = ""
                for chunk in result:
                    if hasattr(chunk, 'text') and chunk.text:
                        buffer += chunk.text
                        sentences = buffer.split('. ')
                        while len(sentences) > 1:
                            sentence = sentences.pop(0)
                            yield sentence + '. '
                        buffer = sentences[0] if sentences else ""
                
                if buffer:
                    yield buffer
                    
        except Exception as e:
            yield f"\n\n⚠️ Error calling Gemini API: {str(e)}\n"
    
    def _extract_csv_from_pdf(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Extract CSV data from PDF using Gemini."""
        chat_interface.add_progress_update("📊 Extracting transaction data...")
        
        csv_prompt = """
        Extract all transactions from this bank statement as CSV with these exact columns:
        Transaction_Date,Posting_Date,Description,Transaction_Type,Merchant_Category,Amount,Location,Balance_After
        
        Rules:
        - Use YYYY-MM-DD format for dates
        - Negative amounts for spending, positive for income
        - Choose appropriate categories: Groceries, Transport, Dining, Retail, Utilities, Entertainment, Health, Cash, Savings, Transfer, Income, Uncategorized
        - Output ONLY the CSV data, no other text
        """
        
        try:
            # Get CSV extraction (non-streaming for parsing)
            csv_data = self._get_gemini_pdf_response(pdf_file, csv_prompt)
            if csv_data:
                df = self._parse_csv_from_response(csv_data)
                chat_interface.set_extracted_data(df)
                yield f"\n\n📊 **Successfully extracted {len(df)} transactions!**"
            else:
                yield "\n\n⚠️ Could not extract transaction data."
        except Exception as e:
            yield f"\n\n⚠️ CSV extraction error: {str(e)}"
    
    def _get_gemini_pdf_response(self, pdf_file, prompt: str) -> Optional[str]:
        """Get a single response from Gemini for PDF (non-streaming)."""
        try:
            if genai_client:
                client = genai_client.Client(api_key=self.config.gemini_api_key)
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(pdf_file.getbuffer())
                    tmp.flush()
                    myfile = client.files.upload(file=tmp.name)
                
                result = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[myfile, "\n\n", prompt]
                )
                
                return getattr(result, "text", None)
                
            elif genai_legacy:
                genai_legacy.configure(api_key=self.config.gemini_api_key)
                model = genai_legacy.GenerativeModel("gemini-2.0-flash-exp")
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(pdf_file.getbuffer())
                    tmp.flush()
                    uploaded_file = genai_legacy.upload_file(path=tmp.name, mime_type="application/pdf")
                
                result = model.generate_content([uploaded_file, "\n\n", prompt])
                
                return getattr(result, "text", None)
                
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return None
    
    def _parse_csv_from_response(self, response: str) -> pd.DataFrame:
        """Parse CSV data from Gemini response."""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            for i, line in enumerate(lines):
                if "Transaction_Date" in line:
                    cleaned = '\n'.join(lines[i:])
                    break
            cleaned = cleaned.strip("`\n ")
        
        df = pd.read_csv(io.StringIO(cleaned))
        
        required_cols = {
            "Transaction_Date", "Posting_Date", "Description", "Transaction_Type",
            "Merchant_Category", "Amount", "Location", "Balance_After"
        }
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns. Got: {list(df.columns)}")
        
        return df
    
    def _fallback_processing(self, pdf_file, chat_interface) -> Generator[str, None, None]:
        """Fallback if PDF processing fails."""
        yield "\n\n📄 Unable to process PDF. Please try again or use demo data."


def create_streaming_processor() -> StreamingGeminiProcessor:
    """Create a new streaming Gemini processor instance."""
    return StreamingGeminiProcessor()