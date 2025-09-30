# AI Spending Analyser

## 🎯 Problem Statement

After researching Trading 212's user base, I've identified that the target users are millennials and Gen Z adults (25-40) who are financially curious but not necessarily financially literate. They're using apps like Monzo and Revolut for banking, they've maybe dabbled in investing through Trading 212, but they still feel anxious about their spending habits. They know they should budget, but traditional budgeting tools feel like homework - too rigid, judgmental, and disconnected from their actual lifestyle. What they really need isn't another app telling them they spend too much on coffee; they need a financial companion that understands their spending is emotional, contextual, and personal.

This app bridges that gap by making spending analysis feel less like a chore and more like having a conversation with a knowledgeable friend who happens to be great with money. One that doesn't just categorise your spending (excel can do that) but it understands that your Friday Pret habit might be your mental health routine, that your weekend Uber rides are how you maintain friendships, and that cutting all "unnecessary" spending isn't realistic or healthy. 

The ultimate value proposition is simple: this app helps users move from financial anxiety to financial awareness without the traditional guilt trip. It's the difference between "You spent £200 on entertainment" and "Your entertainment spending brings you joy and keeps you social - here's how to maintain it while still saving £50/month." This aligns perfectly with Trading 212's mission of democratising finance - to not just making investing accessible, but making financial self-awareness accessible too. This isn't about building another budget tracker - it's about building a spending analyser that actually understands that behind every transaction is a human trying to live their life.

## 🚀 Live Demo

https://spending-analyser.streamlit.app/

## 💡 Key Features & Design Decisions

A couple of [Reddit](https://www.reddit.com/r/trading212/comments/1l8m6gy/listen_to_your_users/) posts have indicated that the current user base highly values simple and intuitive experiences. Therefore, it's imperative that anyone with a Trading 212 account should be able to go from demo data -> value with minimal clicks. 

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

Using Cursor to get setup with boilerplate

Using Claude Code to plan tasks and architectures 

Using Perplexity to research user base and trends to understand users 