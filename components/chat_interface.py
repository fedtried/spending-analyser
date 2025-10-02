"""Chat Interface Component for PDF Bank Statement Processing

Provides a real-time streaming chat experience for PDF analysis with:
- Chat-like UI using st.container() and st.chat_message()
- Message types: user, assistant, system
- Auto-scroll to latest message
- Typing indicator while processing
- Persist chat history in session state
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
import streamlit as st


class ChatMessage:
    """Represents a single chat message with metadata."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        self.role = role  # "user", "assistant", "system"
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatMessage:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp")
        )


class ChatInterface:
    """Manages the chat interface and message history."""
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self) -> None:
        """Initialize chat-related session state variables."""
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # processing_state is now initialized in main app
        
        if "current_pdf" not in st.session_state:
            st.session_state.current_pdf = None
        
        if "extracted_data" not in st.session_state:
            st.session_state.extracted_data = None
        
        if "parsing_metrics" not in st.session_state:
            st.session_state.parsing_metrics = {
                "pages_processed": 0,
                "transactions_found": 0,
                "categories_identified": 0
            }
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the chat history."""
        message = ChatMessage(role, content)
        st.session_state.chat_messages.append(message.to_dict())
    
    def get_messages(self) -> List[ChatMessage]:
        """Get all chat messages."""
        messages = []
        for msg in st.session_state.chat_messages:
            if isinstance(msg, dict):
                messages.append(ChatMessage.from_dict(msg))
            elif isinstance(msg, ChatMessage):
                messages.append(msg)
        return messages
    
    def clear_messages(self) -> None:
        """Clear all chat messages."""
        st.session_state.chat_messages = []
    
    def render_chat_container(self) -> None:
        """Render the main chat container with all messages."""
        messages = self.get_messages()
        
        # Create chat container
        chat_container = st.container()
        
        with chat_container:
            # Render each message
            for message in messages:
                self._render_message(message)
            
            # Show typing indicator if processing
            if st.session_state.processing_state == "streaming":
                self._render_typing_indicator()
    
    def _render_message(self, message: ChatMessage) -> None:
        """Render a single chat message."""
        if message.role == "user":
            with st.chat_message("user"):
                st.write(message.content)
        elif message.role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif message.role == "system":
            with st.chat_message("system"):
                st.info(message.content)
    
    def _render_typing_indicator(self) -> None:
        """Render typing indicator while processing."""
        with st.chat_message("assistant"):
            typing_text = "●●●"
            st.markdown(f"*{typing_text}*")
    
    def update_assistant_message(self, content: str, append: bool = True) -> None:
        """Update the last assistant message or create a new one."""
        if st.session_state.chat_messages and st.session_state.chat_messages[-1].get("role") == "assistant":
            if append:
                # Update the last assistant message
                st.session_state.chat_messages[-1]["content"] += content
            else:
                # Replace the last assistant message
                st.session_state.chat_messages[-1]["content"] = content
        else:
            # Create new assistant message
            self.add_message("assistant", content)
    
    def render_typing_animation(self) -> None:
        """Render a smooth typing animation."""
        import time
        
        typing_frames = ["●●●", "●○○", "○●○", "○○●", "○○○"]
        
        for frame in typing_frames:
            with st.empty():
                st.markdown(f"*{frame}*")
            time.sleep(0.3)
    
    def render_progress_bar(self, current: int, total: int, phase: str) -> None:
        """Render a progress bar for the current phase."""
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{phase} ({current}/{total})")
    
    def add_progress_update(self, message: str) -> None:
        """Add a system progress update message."""
        self.add_message("system", message)
    
    def set_processing_state(self, state: str) -> None:
        """Set the current processing state."""
        st.session_state.processing_state = state
    
    def get_processing_state(self) -> str:
        """Get the current processing state."""
        return st.session_state.processing_state
    
    def set_current_pdf(self, pdf_file) -> None:
        """Set the current PDF file being processed."""
        st.session_state.current_pdf = pdf_file
    
    def get_current_pdf(self):
        """Get the current PDF file."""
        return st.session_state.current_pdf
    
    def set_extracted_data(self, data) -> None:
        """Set the extracted CSV data."""
        st.session_state.extracted_data = data
    
    def get_extracted_data(self):
        """Get the extracted CSV data."""
        return st.session_state.extracted_data
    
    def update_parsing_metrics(self, **kwargs) -> None:
        """Update parsing metrics."""
        for key, value in kwargs.items():
            if key in st.session_state.parsing_metrics:
                st.session_state.parsing_metrics[key] = value
    
    def get_parsing_metrics(self) -> Dict[str, Any]:
        """Get current parsing metrics."""
        return st.session_state.parsing_metrics.copy()
    
    def render_sidebar_metrics(self) -> None:
        """Render parsing metrics in the sidebar."""
        with st.sidebar:
            st.markdown("### Processing Status")
            
            state = self.get_processing_state()
            if state == "idle":
                st.info("Ready to process PDF")
            elif state == "uploading":
                st.info("📤 Uploading PDF...")
            elif state == "streaming":
                st.info("🔄 Analyzing document...")
            elif state == "complete":
                st.success("✅ Analysis complete!")
            
            # Show parsing metrics
            metrics = self.get_parsing_metrics()
            if any(metrics.values()):
                st.markdown("**Progress**")
                st.metric("Pages Processed", metrics.get("pages_processed", 0))
                st.metric("Transactions Found", metrics.get("transactions_found", 0))
                st.metric("Categories Identified", metrics.get("categories_identified", 0))
    
    def render_download_section(self) -> None:
        """Render CSV download section when processing is complete."""
        if self.get_processing_state() == "complete":
            data = self.get_extracted_data()
            if data is not None:
                st.markdown("### 📊 Download Your Data")
                st.caption("Want to explore your data further? Download the full transaction details or use the visualizations below.")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Convert DataFrame to CSV
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="bank_statement_analysis.csv",
                        mime="text/csv",
                        help="Download the extracted transaction data as CSV"
                    )
                
                with col2:
                    # Show preview
                    st.markdown("**Preview (first 5 rows):**")
                    st.dataframe(data.head(), use_container_width=True)
    
    def start_pdf_processing(self, pdf_file) -> None:
        """Initialize PDF processing with initial messages."""
        # Clear previous messages
        self.clear_messages()
        
        # Set current PDF
        self.set_current_pdf(pdf_file)
        
        # Add initial messages
        self.add_message("user", f"📎 Uploaded: {pdf_file.name}")
        self.add_message("assistant", "I can see your bank statement! Let me analyze this for you...")
        self.add_message("system", "📄 PDF received, analyzing document structure...")
        
        # Set processing state
        self.set_processing_state("streaming")
        
        # Reset metrics
        self.update_parsing_metrics(pages_processed=0, transactions_found=0, categories_identified=0)
    
    def start_demo_analysis(self) -> None:
        """Initialize demo data analysis with initial messages."""
        # Clear previous messages
        self.clear_messages()
        
        # Add initial messages
        self.add_message("user", "📊 Selected: Demo Data")
        self.add_message("assistant", "Great! I can see you've selected the demo data. Let me analyze this sample bank statement for you...")
        self.add_message("system", "📄 Demo data loaded, analyzing transaction patterns...")
        
        # Set processing state
        self.set_processing_state("streaming")
        
        # Reset metrics
        self.update_parsing_metrics(pages_processed=1, transactions_found=0, categories_identified=0)
    
    def complete_processing(self, extracted_data) -> None:
        """Complete the processing and set final state."""
        self.set_extracted_data(extracted_data)
        self.set_processing_state("complete")
        
        # No generic completion message - let the AI insights be the natural conclusion
