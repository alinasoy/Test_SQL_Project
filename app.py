"""Ask Your Data - chat with any database or spreadsheet in plain English.  Run: streamlit run app.py"""
import os
import streamlit as st
import core

try:                                   # load a local .env file if present
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")

# ── optional access password for public deployments (set APP_PASSWORD env var / secret) ──
def _secret(name, default=""):
    try:
        return os.getenv(name) or st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

APP_PASSWORD = _secret("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("authed"):
    st.title("💬 Ask Your Data")
    if st.text_input("Password", type="password") == APP_PASSWORD:
        st.session_state.authed = True
        st.rerun()
    st.stop()

ss = st.session_state
for k, v in dict(engine=None, llm=None, schema="", schema_key=None, tables=[], messages=[],
                 suggestions=[], pending=None, source_label="").items():
    ss.setdefault(k, v)


@st.cache_data(ttl=3600, show_spinner=False)
def _detect_models(key):
    return core.list_gemini_models(key)


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.title("💬 Ask Your Data")
    st.caption("Chat with a database or spreadsheet in plain English.")

    st.subheader("1 · Your data")
    source = st.radio("Source", ["Try sample data", "Upload CSV / Excel", "SQLite file", "MySQL", "PostgreSQL"])
    files, db = None, {}
    if source == "Upload CSV / Excel":
        files = st.file_uploader("Files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    elif source == "SQLite file":
        files = st.file_uploader("SQLite database", type=["db", "sqlite", "sqlite3"])
    elif source in ("MySQL", "PostgreSQL"):
        saved_host, saved_pw = _secret("DB_HOST"), _secret("DB_PASSWORD")
        db["host"] = st.text_input("Host", saved_host or "localhost")
        db["port"] = st.number_input("Port", value=int(_secret("DB_PORT") or (3306 if source == "MySQL" else 5432)), step=1)
        db["user"] = st.text_input("User", _secret("DB_USER"))
        typed_pw = st.text_input("Password", type="password", placeholder="(saved password)" if saved_pw else "")
        # saved password is used ONLY for the saved host, so it can never be sent to a different server
        db["password"] = typed_pw or (saved_pw if saved_host and db["host"] == saved_host else "")
        db["database"] = st.text_input("Database", _secret("DB_NAME"))
        st.caption("Tip: use a **read-only** database user.")

    st.subheader("2 · AI model")
    _names = list(core.DEFAULT_MODELS)
    _pref = _secret("LLM_PROVIDER").lower()
    _idx = next((i for i, n in enumerate(_names) if _pref and _pref in n.lower()), 0)
    provider = st.selectbox("Provider", _names, index=_idx)
    env_name = {"Google Gemini": "GOOGLE_API_KEY", "Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY"}[provider]
    server_key = _secret(env_name)
    if server_key and not st.checkbox("Use my own key instead", value=False):
        st.success("🔑 API key loaded from server settings")
        api_key = server_key                       # stays on the server, never sent to the browser
    else:
        api_key = st.text_input("API key", type="password",
                                help="Used only in your browser session. It is never stored or logged.")
    detected = _detect_models(api_key) if provider == "Google Gemini" and api_key else []
    if detected:
        model = st.selectbox("Model (auto-detected for your key)", ["gemini-flash-latest"] + detected,
                             help="gemini-flash-latest always points to Google's newest Flash model, so it never goes out of date.")
    else:
        model = st.text_input("Model", core.DEFAULT_MODELS[provider])
        if provider == "Google Gemini" and api_key:
            st.caption("Could not auto-detect models. Check the key, or type a model name from ai.google.dev/gemini-api/docs/models")

    with st.expander("🛟 Backup models (recommended)"):
        auto_backup = st.checkbox("Auto-fallback to lighter models on the same key", value=True,
                                  help="If the main model is overloaded, the app retries, then switches automatically.")
        b_provider = st.selectbox("Second provider (optional)", ["None"] + list(core.DEFAULT_MODELS))
        b_key = ""
        if b_provider != "None":
            _b_env = {"Google Gemini": "GOOGLE_API_KEY", "Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY"}[b_provider]
            b_key = _secret(_b_env) or st.text_input("Second provider API key", type="password")
            if _secret(_b_env):
                st.caption("🔑 Using the saved key for this provider")

    if st.button("🔌 Connect", type="primary", use_container_width=True):
        try:
            with st.spinner("Connecting..."):
                if source == "Try sample data":
                    engine = core.sample_engine()
                elif source == "Upload CSV / Excel":
                    if not files:
                        raise ValueError("Please upload at least one file.")
                    engine = core.engine_from_files([(f.name, f.getvalue()) for f in files])
                elif source == "SQLite file":
                    if not files:
                        raise ValueError("Please upload a SQLite file.")
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
                    tmp.write(files.getvalue()); tmp.close()
                    engine = core.make_engine("sqlite", path=tmp.name)
                else:
                    engine = core.make_engine(source.lower(), **db)
                if not api_key:
                    raise ValueError("Please enter your AI API key.")
                extra = (b_provider, core.DEFAULT_MODELS[b_provider], b_key) if b_provider != "None" and b_key else None
                ss.llm = core.build_llm(provider, model, api_key, auto_backup, extra,
                                        core.pick_backups(detected, model) if detected else None)
                ss.engine, ss.tables = engine, core.list_tables(engine)
                ss.schema_key, ss.messages, ss.suggestions = None, [], []
                ss.source_label = source
            st.success(f"Connected - {len(ss.tables)} table(s) found")
        except Exception as e:
            st.error(f"Connection failed: {getattr(e, 'orig', e)}")

    if ss.engine is not None:
        st.subheader("3 · Fine-tune (optional)")
        chosen = st.multiselect("Tables to use", ss.tables, default=ss.tables)
        notes = st.text_area("Business notes", placeholder="e.g. Revenue = quantity * unit_price. FY starts in April.")
        if st.button("🗑 Clear chat", use_container_width=True):
            ss.messages = []; st.rerun()


# ───────────────────────── Helpers ─────────────────────────
def looks_like_time(s):
    return bool(s.astype(str).str.match(r"^\d{4}(-\d{2})?(-\d{2})?$").all())

def show_chart(df):
    nums = df.select_dtypes("number").columns.tolist()
    cats = [c for c in df.columns if c not in nums]
    if 2 <= len(df) <= 200 and nums and cats:
        d = df.set_index(cats[0])[nums[:3]]
        (st.line_chart if looks_like_time(df[cats[0]]) else st.bar_chart)(d)

def render(msg, idx):
    if msg["role"] == "user":
        st.markdown(msg["content"]); return
    if msg.get("error"):
        st.error(msg["error"])
        if msg.get("retry") and st.button("🔄 Retry", key=f"retry{idx}"):
            ss.messages = ss.messages[:idx - 1]          # drop the failed turn and ask again
            ss.pending = msg["retry"]; st.rerun()
        return
    st.markdown(msg["answer"])
    if "(backup)" in msg.get("model", ""):
        st.caption(f"ℹ️ Main model was busy - answered by {msg['model']}")
    df = msg["df"]
    t1, t2, t3 = st.tabs(["📋 Table", "📈 Chart", "🧾 SQL"])
    with t1:
        st.dataframe(df, use_container_width=True)
        if msg["truncated"]:
            st.caption("Showing the first rows only.")
        st.download_button("⬇ Download CSV", df.to_csv(index=False), f"result_{idx}.csv", "text/csv", key=f"dl{idx}")
    with t2:
        show_chart(df) if not df.empty else st.info("No data to chart.")
    with t3:
        st.code(msg["sql"], language="sql")
        if len(msg["attempts"]) > 1:
            st.caption(f"Auto-corrected after {len(msg['attempts']) - 1} failed attempt(s).")


# ───────────────────────── Main ─────────────────────────
if ss.engine is None:
    st.header("Ask questions about your data - no SQL needed")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Add data**\n\nUpload CSV/Excel, or connect MySQL / PostgreSQL / SQLite.")
    c2.markdown("**2. Add an AI key**\n\nGoogle Gemini (free tier), Groq or OpenAI.")
    c3.markdown("**3. Ask**\n\nGet answers, tables, charts and the SQL behind them.")
    st.info("New here? Choose **Try sample data**, paste an API key, and press **Connect**.")
    st.stop()

# (re)build the schema text when the table selection changes
key = tuple(chosen)
if ss.schema_key != key:
    ss.schema = core.describe_schema(ss.engine, list(chosen))
    ss.schema_key = key
    if not ss.suggestions:
        try:
            ss.suggestions = core.suggest_questions(ss.llm, ss.schema)
        except Exception:
            ss.suggestions = []

with st.expander(f"📚 Schema ({len(chosen)} tables)"):
    st.code(ss.schema, language="text")

for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        render(m, i)

if not ss.messages and ss.suggestions:
    st.caption("Try one of these:")
    cols = st.columns(len(ss.suggestions))
    for col, s in zip(cols, ss.suggestions):
        if col.button(s, use_container_width=True):
            ss.pending = s; st.rerun()

question = st.chat_input("Ask a question about your data...") or ss.pending
ss.pending = None
if question:
    ss.messages.append({"role": "user", "content": question})
    history = "\n".join(f"Q: {m['content']}" + (f" -> {ss.messages[j+1].get('sql','')}"
                        if j + 1 < len(ss.messages) - 1 else "")
                        for j, m in enumerate(ss.messages[:-1]) if m["role"] == "user")[-1500:]
    try:
        with st.spinner("Thinking..."):
            sql, df, truncated, attempts = core.text_to_sql(ss.llm, ss.engine, question, ss.schema, notes, history)
            answer = core.explain_result(ss.llm, question, sql, df)
        ss.messages.append({"role": "assistant", "answer": answer, "sql": sql, "df": df,
                            "truncated": truncated, "attempts": attempts, "model": ss.llm.last_used})
    except core.UnsafeSQLError as e:
        ss.messages.append({"role": "assistant", "error": f"🚫 Blocked for safety: {e}"})
    except core.LLMUnavailableError as e:
        ss.messages.append({"role": "assistant", "error": f"⏳ {e}", "retry": question})
    except Exception as e:
        ss.messages.append({"role": "assistant", "error": f"Sorry, that didn't work: {e}", "retry": question})
    st.rerun()
