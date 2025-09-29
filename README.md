# AI Spending Analyser

## 🎯 Problem Statement
[User problem, why it matters, target audience]

## 🚀 Live Demo
[Link with demo credentials]

## 💡 Key Features & Design Decisions
[Explain each feature and WHY you built it]

## 🏗️ Technical Architecture
[Simple diagram, tech choices with rationale]

## 📊 Product Metrics
[What you'd measure and why]

## 🔮 Future Vision
[6-month roadmap if this were real]

## 🤝 Cross-functional Considerations
[Legal, compliance, support, sales angles]

## 💻 Local Development

### Prerequisites
- Python 3.9+ installed on your system
- Git for version control

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
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

5. **View the app**
   - Open your browser to `http://localhost:8501`
   - The app will automatically reload when you make changes

### Development Workflow

- **Activate venv**: Always run `source .venv/bin/activate` before working
- **Install new packages**: Add to `requirements.txt` and run `pip install -r requirements.txt`
- **Deactivate venv**: Run `deactivate` when finished

### Project Structure
```
spending-analyser/
├── .venv/                    # Virtual environment (don't commit)
├── .streamlit/
│   ├── config.toml          # Dark theme configuration
│   └── secrets.example.toml # API keys template
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── data/                   # Sample data directory
├── utils/                  # Utility functions
└── components/             # Reusable UI components
```

### Environment Variables
- Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
- Add your API keys (OpenAI, etc.) to `secrets.toml`
- Never commit `secrets.toml` to version control

## 📝 Development Journal
[Link to commits showing iterative process]