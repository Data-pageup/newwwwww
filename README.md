# 🧠 GenAI Tools

AI-powered health and career tools built with **Google Gemini 2.5 Flash**, **LangChain**, and **Streamlit**.

---

## 📦 Projects

### 🩸 BloodWork Analyzer
Upload your blood test report and get:
- Extracted test values classified as `HIGH` / `LOW` / `NORMAL` with severity ratings
- A personalized **Indian diet plan** — foods to eat, avoid, and a sample daily meal structure
- Downloadable analysis report

### 📊 Resume Analyzer
Upload your resume PDF and get:
- Honest scoring (0–10) against **4 data roles** — Data Scientist, Data Analyst, ML Engineer, AI Engineer
- Brief reasoning for each score
- **3 actionable improvement suggestions** from an AI career mentor

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Web UI |
| `LangChain` | LLM orchestration |
| `Google Gemini 2.5 Flash` | AI model |
| `pdfplumber` | PDF text extraction |
| `python-dotenv` | Environment variable management |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/genai-health-tools.git
cd genai-health-tools
```

### 2. Install dependencies

```bash
pip install streamlit langchain-google-genai pdfplumber python-dotenv
```

### 3. Set up your API key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the apps

```bash
# BloodWork Analyzer
streamlit run bloodwork_analyzer/app.py

# Resume Analyzer
streamlit run resume_analyzer/app.py
```

---

---

## ⚠️ Disclaimer

These tools are for **informational purposes only**.

- **BloodWork Analyzer** — Not a substitute for professional medical advice. Always consult your doctor before making dietary or health changes.
- **Resume Analyzer** — AI-generated feedback. Use it as a guide, not a guarantee.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
