<<<<<<< HEAD
# Ask Your Data
Chat with MySQL / PostgreSQL / SQLite databases or CSV/Excel files in plain English.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy so everyone can use it
- **Streamlit Community Cloud** (free): push this folder to GitHub -> share.streamlit.io -> New app -> `app.py`.
- **Hugging Face Spaces** (free): create a Streamlit Space and upload the files.
- **Docker / any cloud**: `docker build -t askdata . && docker run -p 8501:8501 askdata`

## Settings (env vars or Streamlit secrets, all optional)
- `APP_PASSWORD` - require a password before the app opens
- `GOOGLE_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` - pre-fill the key (otherwise each user pastes their own)

## Security notes
- Only single read-only SELECT/WITH queries are executed. Still use a **read-only DB user**.
- API keys live only in the user's session. Nothing is stored or logged.
- A cloud-hosted app can only reach databases that are reachable from the internet.

## Save your API key once (no more pasting)
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your key.
   (Alternative: copy `.env.example` to `.env`.)
2. Restart the app. The sidebar shows "API key loaded from server settings".
3. On Streamlit Community Cloud: App -> Settings -> Secrets -> paste the same lines.
`.gitignore` already excludes both files so your keys never reach GitHub.
=======
# Test_SQL_Project
>>>>>>> b038efe9c51c23dab1d2c1cf2a9e86123e190eb6
