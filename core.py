"""Core Text-to-SQL engine: works with MySQL, PostgreSQL, SQLite and uploaded CSV/Excel files."""
import io, re, os, time, random
import pandas as pd
from sqlalchemy import create_engine, inspect, text, URL
from sqlalchemy.pool import StaticPool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


# ───────────────────────── LLM factory ─────────────────────────
DEFAULT_MODELS = {
    "Google Gemini": "gemini-flash-latest",   # alias: Google keeps it pointed at the newest Flash
    "Groq": "llama-3.3-70b-versatile",
    "OpenAI": "gpt-4o-mini",
}

# Lighter / separate-capacity models on the SAME key, tried automatically if the main model is overloaded.
AUTO_BACKUP_MODELS = {
    "Google Gemini": ["gemini-flash-lite-latest", "gemini-flash-latest"],
    "Groq": ["llama-3.1-8b-instant"],
    "OpenAI": ["gpt-4.1-mini"],
}

def get_llm(provider: str, model: str, api_key: str):
    # fail fast (short timeout, 1 internal retry) so OUR retry/fallback logic controls the waiting
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0,
                                      timeout=30, max_retries=1)
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0, timeout=30, max_retries=1)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=0, timeout=30, max_retries=1)
    raise ValueError(f"Unknown provider: {provider}")


# ───────────────────────── Model auto-detection (survives model retirements) ─────────────────────────
def _parse_gemini_models(payload: dict):
    """Keep stable text models that support generateContent, newest version first (flash before flash-lite)."""
    found = []
    for m in payload.get("models", []):
        name = m.get("name", "").replace("models/", "")
        hit = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-(flash|flash-lite)", name)
        if hit and "generateContent" in m.get("supportedGenerationMethods", []):
            found.append((float(hit.group(1)), hit.group(2) == "flash", name))
    return [n for _, _, n in sorted(found, reverse=True)]

def list_gemini_models(api_key: str):
    """Ask Google which Gemini models this key can use right now. Returns [] if the call fails."""
    import json, urllib.request
    try:
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
            headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=10) as r:
            return _parse_gemini_models(json.load(r))
    except Exception:
        return []

def pick_backups(detected, model, n=2):
    """Prefer lite models (separate capacity), then other flash models, never the main one."""
    others = [m for m in detected if m != model]
    return ([m for m in others if m.endswith("-lite")] + [m for m in others if not m.endswith("-lite")])[:n]

# ───────────────────────── Resilience: retry + fallback ─────────────────────────
class LLMUnavailableError(Exception):
    """Every configured model failed (busy, rate-limited, unreachable or misconfigured)."""

_TRANSIENT = ("503", "429", "500", "502", "504", "unavailable", "overloaded", "high demand",
              "resource_exhausted", "resource exhausted", "rate limit", "quota", "timeout",
              "timed out", "deadline", "temporarily", "connection", "try again", "internal")
_CONFIG = ("api key", "api_key", "401", "403", "permission", "unauthorized", "invalid", "not found", "404")

def is_transient(e: Exception) -> bool:
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(code, int):
        return code in (408, 409, 429, 500, 502, 503, 504)
    msg = str(e).lower()
    return any(m in msg for m in _TRANSIENT) and not any(m in msg for m in ("api key", "permission"))

class ResilientLLM(Runnable):
    """Drop-in LLM: retries transient errors with exponential backoff + jitter, then falls back
    to the next model. Use like a normal model:  prompt | llm | parser"""
    def __init__(self, models, retries=3, base_delay=1.5, max_delay=10):
        self.models, self.retries = models, retries
        self.base_delay, self.max_delay = base_delay, max_delay
        self.last_used, self.last_errors = "", []

    def invoke(self, input, config=None, **kwargs):
        self.last_errors = []
        for label, model in self.models:
            for attempt in range(self.retries):
                try:
                    out = model.invoke(input)
                    self.last_used = label
                    return out
                except Exception as e:
                    self.last_errors.append(f"{label}: {str(e)[:160]}")
                    if not is_transient(e):
                        break                                   # retrying won't help -> next model
                    if attempt < self.retries - 1:
                        time.sleep(min(self.base_delay * 2 ** attempt, self.max_delay) + random.random())
        errs = " ".join(self.last_errors).lower()
        if any(m in errs for m in ("no longer available", "not found", "404")):
            hint = "The selected AI model is retired or the name is wrong. Choose another model in the sidebar and press Connect."
        elif any(m in errs for m in _CONFIG) and not any(m in errs for m in ("503", "429", "overloaded", "unavailable")):
            hint = "Please check your API key and model name."
        else:
            hint = "The AI service is busy or unreachable right now. Your data connection is fine - please retry in a moment."
        raise LLMUnavailableError(f"{hint}\n\nDetails: " + " | ".join(self.last_errors[-3:]))

def build_llm(provider, model, api_key, auto_backup=True, extra=None, backup_models=None):
    """extra = (provider, model, api_key) of an optional second provider used as a last resort."""
    models = [(f"{provider}:{model}", get_llm(provider, model, api_key))]
    if auto_backup:
        names = list(backup_models or AUTO_BACKUP_MODELS.get(provider, []))
        if provider == "Google Gemini":                 # aliases never retire -> permanent safety net
            names += [m for m in AUTO_BACKUP_MODELS["Google Gemini"] if m not in names]
        for m in dict.fromkeys(names):                  # de-duplicate, keep order
            if m != model:
                try:
                    models.append((f"{provider}:{m} (backup)", get_llm(provider, m, api_key)))
                except Exception:
                    pass
    if extra and extra[2]:
        try:
            models.append((f"{extra[0]}:{extra[1]} (backup)", get_llm(*extra)))
        except Exception:
            pass
    return ResilientLLM(models)


# ───────────────────────── Connections ─────────────────────────
def make_engine(kind: str, host="", port=None, user="", password="", database="", path=""):
    if kind == "mysql":
        url = URL.create("mysql+pymysql", user, password, host, int(port or 3306), database)
    elif kind == "postgresql":
        url = URL.create("postgresql+psycopg2", user, password, host, int(port or 5432), database)
    elif kind == "sqlite":
        url = f"sqlite:///{path}"
    else:
        raise ValueError(kind)
    args = {} if kind == "sqlite" else {"connect_timeout": 10}
    engine = create_engine(url, pool_pre_ping=True, connect_args=args)
    with engine.connect() as c:           # fail fast with a clear error
        c.execute(text("SELECT 1"))
    return engine

def _slug(name: str, fallback="table") -> str:
    s = re.sub(r"\W+", "_", str(name).strip()).strip("_").lower() or fallback
    return f"t_{s}" if s[0].isdigit() else s

def engine_from_files(files):
    """files = [(filename, bytes)]. Every CSV / Excel sheet becomes a table in an in-memory SQLite DB."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    for fname, data in files:
        stem, ext = os.path.splitext(os.path.basename(fname))
        if ext.lower() == ".csv":
            frames = {stem: pd.read_csv(io.BytesIO(data))}
        elif ext.lower() in (".xlsx", ".xls"):
            frames = pd.read_excel(io.BytesIO(data), sheet_name=None)
            if len(frames) == 1:
                frames = {stem: next(iter(frames.values()))}
            else:
                frames = {f"{stem}_{k}": v for k, v in frames.items()}
        else:
            raise ValueError(f"Unsupported file type: {fname}")
        for name, df in frames.items():
            df.columns = [_slug(c, "col") for c in df.columns]      # clean column names help the LLM
            df.to_sql(_slug(name), engine, index=False, if_exists="replace")
    return engine

def sample_engine():
    """Tiny demo database so anyone can try the app instantly."""
    import random
    random.seed(7)
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    customers = pd.DataFrame({"customer_id": range(1, 21),
                              "name": [f"Customer {i}" for i in range(1, 21)],
                              "country": random.choices(["India", "USA", "Germany", "UK"], k=20)})
    products = pd.DataFrame({"product_id": range(1, 9),
                             "product_name": [f"Product {i}" for i in range(1, 9)],
                             "category": ["Phones", "Phones", "Laptops", "Laptops", "Audio", "Audio", "Tablets", "Tablets"],
                             "unit_price": [500, 650, 1200, 1500, 80, 150, 400, 550]})
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for oid in range(1, 401):
        p = products.sample(1, random_state=oid).iloc[0]
        q = random.randint(1, 5)
        rows.append({"order_id": oid, "order_date": str(random.choice(dates).date()),
                     "customer_id": random.randint(1, 20), "product_id": int(p.product_id),
                     "quantity": q, "revenue": float(p.unit_price * q)})
    for name, df in {"customers": customers, "products": products, "orders": pd.DataFrame(rows)}.items():
        df.to_sql(name, engine, index=False)
    return engine


# ───────────────────────── Schema ─────────────────────────
def list_tables(engine):
    insp = inspect(engine)
    return sorted(insp.get_table_names() + insp.get_view_names())

def describe_schema(engine, tables=None, sample_rows=2) -> str:
    insp, q = inspect(engine), engine.dialect.identifier_preparer.quote
    parts = []
    for t in (tables or list_tables(engine)):
        cols = ", ".join(f"{q(c['name'])} {c['type']}" for c in insp.get_columns(t))
        block = f"TABLE {q(t)} ({cols})"
        for fk in insp.get_foreign_keys(t):
            block += f"\n  FOREIGN KEY {fk['constrained_columns']} -> {fk['referred_table']}{fk['referred_columns']}"
        try:
            with engine.connect() as c:
                s = pd.read_sql_query(text(f"SELECT * FROM {q(t)} LIMIT {sample_rows}"), c)
            s = s.astype(str).apply(lambda col: col.str.slice(0, 40))
            block += "\n  Sample rows:\n  " + s.to_string(index=False).replace("\n", "\n  ")
        except Exception:
            pass
        parts.append(block)
    return "\n\n".join(parts)


# ───────────────────────── Safety ─────────────────────────
class UnsafeSQLError(Exception):
    pass

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|grant|revoke|merge|call|exec|execute|"
    r"attach|detach|pragma|vacuum|copy|load_file|sleep|pg_sleep|benchmark)\b|into\s+(outfile|dumpfile)", re.I)

def clean_sql(raw) -> str:
    t = str(raw).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1)
    t = re.sub(r"^\s*sql(\s*query)?\s*:\s*", "", t, flags=re.I)
    return " ".join(t.split()).strip()

def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQLError("Empty query.")
    if ";" in s:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQLError("Only read-only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise UnsafeSQLError("The query contains a forbidden keyword.")
    return s


# ───────────────────────── Execution ─────────────────────────
def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df

def run_sql(engine, sql: str, max_rows=1000):
    sql = validate_sql(sql)
    safe = re.sub(r"(?<!\\):", r"\\:", sql)          # stop SQLAlchemy treating ':' as a bind param
    with engine.connect() as conn:
        res = conn.execute(text(safe))
        rows = res.fetchmany(max_rows + 1)
        df = pd.DataFrame(rows[:max_rows], columns=list(res.keys()))
    return sql, _coerce_numeric(df), len(rows) > max_rows


# ───────────────────────── LLM chains ─────────────────────────
SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior data analyst. Write ONE read-only {dialect} query that answers the question.
Rules:
- Output ONLY the SQL. No markdown, no explanation, a single statement.
- Only SELECT or WITH ... SELECT. Never modify data.
- Use only the tables and columns in the schema. Quote identifiers exactly as shown in the schema.
- Use {dialect}-compatible syntax (date functions, string functions, LIMIT).
- Add LIMIT 100 unless the query is an aggregate returning few rows.

Schema:
{schema}

Business notes from the user (may be empty):
{notes}"""),
    ("human", "Previous turns:\n{history}\n\nQuestion: {question}\n{feedback}\nSQL:"),
])
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}
SQL used: {sql}
Result (first rows):
{result}

Answer the question in 1-3 plain sentences using ONLY the result. Format numbers nicely.
If the result is empty, say that no data was found.""")
SUGGEST_PROMPT = ChatPromptTemplate.from_template(
    """Given this database schema, write 4 short, useful questions a business user might ask.
Return one question per line, no numbering, no extra text.

{schema}""")

def text_to_sql(llm, engine, question, schema, notes="", history="", retries=3, max_rows=1000):
    """Generate -> validate -> run. On a DB error, feed the error back to the LLM and retry."""
    chain = SQL_PROMPT | llm | StrOutputParser()
    feedback, attempts = "", []
    for _ in range(retries):
        raw = clean_sql(chain.invoke({
            "dialect": engine.dialect.name, "schema": schema, "notes": notes or "(none)",
            "history": history or "(none)", "question": question, "feedback": feedback}))
        try:
            sql, df, truncated = run_sql(engine, raw, max_rows)
            attempts.append({"sql": raw, "error": None})
            return sql, df, truncated, attempts
        except UnsafeSQLError:
            raise
        except Exception as e:
            err = str(getattr(e, "orig", e))[:400]
            attempts.append({"sql": raw, "error": err})
            feedback = f"Your previous query failed.\nQuery: {raw}\nDatabase error: {err}\nRewrite it so it runs."
    raise RuntimeError(f"Could not build a working query. Last error: {attempts[-1]['error']}")

def explain_result(llm, question, sql, df) -> str:
    """The data is already fetched - if the AI summary fails, still show the table."""
    try:
        return (ANSWER_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question, "sql": sql, "result": df.head(20).to_string(index=False)}).strip()
    except Exception:
        return f"Here are the results ({len(df)} row(s)). The AI summary is temporarily unavailable."

def suggest_questions(llm, schema):
    out = (SUGGEST_PROMPT | llm | StrOutputParser()).invoke({"schema": schema})
    return [l.strip("-•* ").strip() for l in out.splitlines() if l.strip()][:4]
