# AI Spending Analyser

## 🎯 Problem Statement

Target users are financially curious millennials and Gen Z (25–40) who use Monzo/Revolut and may dabble in Trading 212, but feel anxious about spending and find traditional budgeting rigid and judgmental. This app makes analysis conversational and empathetic—going beyond categorisation to recognise context (e.g., a Friday Pret as self‑care) and provide realistic, non‑shaming guidance. The goal is to move users from anxiety to awareness with actionable insights that fit their life, aligning with Trading 212’s mission to democratise financial self‑awareness.

## 🚀 Live Demo

https://spending-analyser.streamlit.app/

## 💡 Key Features & Design Decisions

A couple of [Reddit](https://www.reddit.com/r/trading212/comments/1l8m6gy/listen_to_your_users/) posts have indicated that the current user base highly values simple and intuitive experiences. Therefore, it's imperative that anyone with a Trading 212 account should be able to go from demo data -> value with minimal clicks. 

Natural Language analysis where the prompt considers the audience, what they value and the insight they'd be looking to gain from this tool. 

> For example, if someone is spending 50% of their income on rent in London that's about average and there's no point telling them to reduce their spending. However, if they're spending money on takeaways AND grocery shopping it may be a good idea to suggest meal planning to reduce food waste, and the cost associated.  

Natural language feature is also in a chat style with streaming responses as we have got used to this method of talking with AI bots. It seems more personable too. 

The main graph is a time series visualisation of spending vs income with main balance overlayed ontop. This is useful for users so they can match up high spending periods with a particular event. It also makes it really easy to see the 'drip' spending - minor but every single day. This makes it easier for people to spot visual patterns that raw numbers from a bank statement can't provide. 

## 🏗️ Technical Architecture
The app is a Streamlit client that keeps all user state in memory. When a CSV (or demo data) is provided, a lightweight ingestion layer detects the schema, normalizes fields (date, merchant, category, amount), validates with Pydantic, and builds derived tables and metrics in Pandas. These same frames power both the Plotly visuals and the context packaged for the AI.

For analysis, a structured prompt (user intent + summarized tables + guardrails) is sent to Gemini 2.0 Flash and responses are streamed directly into the chat UI. Errors or timeouts fall back to deterministic summaries so the interface remains responsive.

Key considerations:
- Reliability: schema detection with sensible defaults, strict validation, and graceful fallbacks for AI/service failures.
- Performance: all processing is in-memory, incremental filtering on precomputed frames, streaming-first UX to reduce perceived latency.
- Security/Privacy: no data is persisted; secrets are read from Streamlit `secrets.toml`; uploads stay in session memory.
- Accessibility/UX: Plotly charts reflect current filters; state-driven UI avoids recomputation beyond what changed.

## 💻 Local Development

### Prerequisites
- Python 3.9+ installed on your system
- Git for version control

### Quick Start

1. **Clone the repository**
   ```bash
   git clone git@github.com:fedtried/spending-analyser.git
   cd spending-analyser
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   
### Environment Variables
- Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
- Add your API keys (OpenAI, etc.) to `secrets.toml`
- Never commit `secrets.toml` to version control
