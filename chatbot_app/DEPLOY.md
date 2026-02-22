# Deploy Chatbot Free on Streamlit Cloud

## Folder structure expected on GitHub

```
your-repo/
├── my_agent/               ← agent framework
│   ├── agent/
│   ├── llm/
│   ├── tools/
│   ├── memory/
│   └── config.py
├── chatbot_app/            ← this app
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
└── .gitignore
```

## Step-by-step deployment

### 1. Push to GitHub

```bash
cd /Users/rakshitsakhuja

git init
git add my_agent/ chatbot_app/ .gitignore
git commit -m "initial commit"

# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to → https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - Repository: `your-username/your-repo`
   - Branch: `main`
   - Main file path: `chatbot_app/app.py`    ← important
5. Click **"Deploy"**

### 3. Add your API key (Secrets)

In Streamlit Cloud dashboard:
1. Click your app → **Settings** → **Secrets**
2. Paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

3. Save → app restarts automatically

Done. Your chatbot is live at:
`https://your-app-name.streamlit.app`

---

## Run locally

```bash
cd /Users/rakshitsakhuja/chatbot_app

# Copy secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API key

pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501

---

## What users see

- Clean chat interface with dark theme
- Sidebar to pick provider / model / persona
- Starter questions on empty state
- Tool usage shown in expandable section
- Chat history persists within session
