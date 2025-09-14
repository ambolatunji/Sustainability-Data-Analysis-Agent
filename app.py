from __future__ import annotations
import io
import pickle
import re
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import faiss

from core.ingestion import ingest_files, IngestedData
from core.vectorstore import build_faiss_index, VectorIndex
from core.sql_assistant import SQLAssistant
from core.llm_client import LLMClient
from core.forecaster import Forecaster
from prompts import system_prompt, sql_prompt, answer_prompt
from core.utils import parse_date_series


# Optional web search
try:
    from duckduckgo_search import DDGS
    _HAS_DDG = True
except Exception:
    _HAS_DDG = False

st.set_page_config(page_title="Sustainability Copilot", layout="wide")
st.title("🌍 Sustainability Copilot for Manufacturing")
st.caption("Upload ESG data (Excel/CSV/PDF), ask questions, run SQL calculations, and forecast.")

# --- Session state init ---
if 'vector' not in st.session_state:
    st.session_state.vector: Optional[VectorIndex] = None
if 'ingested' not in st.session_state:
    st.session_state.ingested: Optional[IngestedData] = None
if 'sql' not in st.session_state:
    st.session_state.sql: Optional[SQLAssistant] = None

# --- Sidebar: model selection ---
import os
with st.sidebar:
    def ok(name): #return "✅" if os.getenv(name) else "❌"
        try:
            import streamlit as st
            v = st.secrets.get("api", {}).get(name) or st.secrets.get(name)
        except Exception:
                v = None
        v = v or os.getenv(name)
        return "✅" if v else "❌"
    st.write(f"Keys → OpenAI {ok('OPENAI_API_KEY')} · Groq {ok('GROQ_API_KEY')} · DeepSeek {ok('DEEPSEEK_API_KEY')}")
    st.header("Models & Options")
    provider = st.selectbox("LLM Provider", ["openai", "groq", "deepseek", "disabled"], index=0)
    model_name = st.text_input(
        "Model name",
        value=(
            "gpt-4o-mini" if provider=="openai" else
            "llama-3.1-70b-versatile" if provider=="groq" else
            "deepseek-reasoner" if provider=="deepseek" else ""
        )
    )

    use_rag = st.checkbox("Use RAG context", value=True)
    top_k = st.slider("RAG passages (k)", 0, 10, 4)
    sql_rows = st.slider("SQL rows passed to LLM", 50, 2000, 400, step=50)
    use_web = st.checkbox("Augment with web if needed", value=True)
    st.divider()
    save_btn = st.button("💾 Save index to disk")
    load_btn = st.button("📂 Load index from disk")

# --- File upload & index build ---
uploaded = st.file_uploader("Upload Excel/CSV/PDF", type=["xlsx","xlsm","xls","csv","pdf"], accept_multiple_files=True)
if st.button("Build/Refresh Index", type="primary"):
    if not uploaded:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Parsing files and building index..."):
            ing = ingest_files(uploaded)
            st.session_state.ingested = ing
            vec = build_faiss_index(ing.corpus_docs)
            st.session_state.vector = vec
            st.session_state.sql = SQLAssistant(ing.tables)
        st.success(f"Indexed {len(ing.corpus_docs)} chunks from {len(ing.tables)} tables.")

# --- Save/Load index ---
if save_btn:
    if st.session_state.vector is None or st.session_state.ingested is None:
        st.warning("Nothing to save yet.")
    else:
        blob_tables = {}
        for k, v in st.session_state.ingested.tables.items():
            buf = io.BytesIO()
            v.to_parquet(buf, index=False)
            blob_tables[k] = buf.getvalue()
        blob = {
            'vector': {
                'ids': st.session_state.vector.ids,
                'embeddings': st.session_state.vector.embeddings,
                'model_name': st.session_state.vector.model_name,
            },
            'docs': st.session_state.ingested.corpus_docs,
            'tables': blob_tables,
        }
        with open('sustainability_index.pkl', 'wb') as f:
            pickle.dump(blob, f)
        st.success("Saved → sustainability_index.pkl")

if load_btn:
    try:
        with open('sustainability_index.pkl', 'rb') as f:
            blob = pickle.load(f)
        X = blob['vector']['embeddings']
        dim = X.shape[1] if X.size else 384
        index = faiss.IndexFlatIP(dim)
        if X.size:
            index.add(X.astype('float32'))
        st.session_state.vector = VectorIndex(index=index, ids=blob['vector']['ids'], embeddings=X, model_name=blob['vector']['model_name'])
        tables = {k: pd.read_parquet(io.BytesIO(pq)) for k, pq in blob['tables'].items()}
        st.session_state.ingested = IngestedData(tables=tables, corpus_docs=blob['docs'])
        st.session_state.sql = SQLAssistant(tables)
        st.success("Index loaded from disk.")
    except Exception as e:
        st.error(f"Failed to load: {e}")


# --- Catalog view ---
if st.session_state.sql:
    with st.expander("Catalog (tables & columns)", expanded=False):
        st.code(st.session_state.sql.describe())

st.divider()

# --- Catalog Explorer ---
st.subheader("📚 Catalog Explorer")
if st.session_state.ingested and st.session_state.ingested.tables:
    tname = st.selectbox("Table", list(st.session_state.ingested.tables.keys()))
    df = st.session_state.ingested.tables[tname]
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    col = st.selectbox("Column (optional)", ["<all>"] + list(df.columns))
    if col == "<all>":
        st.dataframe(df.head(100))
    else:
        st.write("Summary:")
        st.dataframe(pd.DataFrame({
            'non_null': [df[col].notna().sum()],
            'dtype': [str(df[col].dtype)],
            'unique': [df[col].nunique()]
        }))
        st.dataframe(df[[col]].head(200))
else:
    st.info("Upload data and build the index to explore the catalog.")

st.divider()
# --- QA / RAG panel ---
st.subheader("Ask a question ✨ (RAG + SQL + optional Web)")
q = st.text_area("Question", placeholder="e.g., What was total GHG emissions by plant in 2024 Q1? Compare to Q1 2023.")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    run_rag = st.button("🔎 Retrieve & Answer", type="primary")
with c2:
    gen_sql = st.button("🧮 Propose SQL")
with c3:
    exec_sql = st.button("▶️ Execute SQL & Answer")

answer_box = st.empty()

# --- Web search helper ---
@st.cache_data(show_spinner=True)
def web_search_brief(query: str, max_results: int = 4) -> str:
    if not _HAS_DDG:
        return "(Web search unavailable)"
    notes = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get('title') or ''
                href = r.get('href') or ''
                body = r.get('body') or ''
                notes.append(f"- {title}: {body} ({href})")
    except Exception:
        return "(Web search failed)"
    return "\n".join(notes[:max_results])


def maybe_web_notes(question: str, local_hit_count: int, use_web: bool) -> str:
    if not use_web:
        return ""
    if local_hit_count < 2 or any(w in question.lower() for w in ["benchmark","standard","policy","target","net zero","scope 3"]):
        return web_search_brief(question)
    return ""

# --- LLM helpers ---

def propose_sql(question: str, sql_asst: SQLAssistant, llm: Optional[LLMClient]) -> str:
    catalog = sql_asst.describe()
    if llm is None:
        # Fallback heuristic
        for tname, df in st.session_state.ingested.tables.items():
            if 'Date' in df.columns and df.select_dtypes(include='number').shape[1] > 0:
                numcol = df.select_dtypes(include='number').columns[0]
                safe = sql_asst._safe_name(tname)
                return f"SELECT Date, SUM({numcol}) AS value FROM {safe} GROUP BY 1 ORDER BY 1;"
        return "SELECT 'NO_SQL';"

    prompt = sql_prompt.format(question=question, catalog=catalog)
    try:
        sql = llm.chat([
            {"role": "system", "content": "You only return SQL code."},
            {"role": "user", "content": prompt},
        ], temperature=0.0, max_tokens=400)
        m = re.search(r"```sql\n(.*?)```", sql, flags=re.S|re.I)
        if m:
            sql = m.group(1).strip()
        return sql.strip().rstrip(';') + ';'
    except Exception as e:
        st.warning(f"SQL proposal failed: {e}")
        return "SELECT 'NO_SQL';"


def answer_with_context(question: str, passages: List[Tuple[str,float]], sql_df: Optional[pd.DataFrame], web_notes: str, llm: Optional[LLMClient]) -> str:
    context = ""
    if use_rag and st.session_state.ingested:
        ctx_parts = []
        for pid, _ in passages[:5]:
            for (doc_id, txt) in st.session_state.ingested.corpus_docs:
                if doc_id == pid:
                    ctx_parts.append(f"[{pid}] {txt[:1800]}")
                    break
        context = "\n\n".join(ctx_parts)

    sql_result = ""
    if sql_df is not None and not sql_df.empty:
        sql_result = sql_df.head(800).to_csv(index=False)

    content = answer_prompt.format(
        system=system_prompt,
        context=context,
        sql_result=sql_result,
        web=web_notes or "",
        question=question,
    )

    if llm is None:
        return "(LLM disabled)\n\n" + (context[:800] or "No local context found.")

    try:
        ans = llm.chat([
            {"role": "system", "content": "You are a precise, concise sustainability analyst."},
            {"role": "user", "content": content},
        ], temperature=0.2, max_tokens=800)
        return ans
    except Exception as e:
        return f"Answer generation failed: {e}"

# --- Drive buttons ---
llm = None if provider == 'disabled' else LLMClient(provider, model_name)

if run_rag:
    if not st.session_state.vector or not st.session_state.ingested:
        answer_box.error("Please upload data and build the index first.")
    else:
        hits = st.session_state.vector.search(q, k=top_k) if (use_rag and top_k>0) else []
        web_notes = maybe_web_notes(q, len(hits), use_web)
        ans = answer_with_context(q, hits, None, web_notes, llm)
        answer_box.markdown(ans)
        if hits:
            with st.expander("Retrieved chunks"):
                for pid, score in hits:
                    st.write(f"**{pid}** (score={score:.3f})")

if gen_sql:
    if not st.session_state.sql:
        st.warning("Build the index first (loads tables).")
    else:
        sql = propose_sql(q, st.session_state.sql, llm)
        st.code(sql, language='sql')

if exec_sql:
    if not st.session_state.sql:
        st.warning("Build the index first (loads tables).")
    else:
        sql = propose_sql(q, st.session_state.sql, llm)
        st.code(sql, language='sql')
        try:
            df = st.session_state.sql.run(sql)
            st.dataframe(df)
        except Exception as e:
            st.error(f"SQL error: {e}")
            df = None
        hits = st.session_state.vector.search(q, k=top_k) if (use_rag and st.session_state.vector and top_k>0) else []
        web_notes = maybe_web_notes(q, len(hits), use_web)
        ans = answer_with_context(q, hits, df, web_notes, llm)
        answer_box.markdown(ans)

st.divider()

# --- Forecast panel ---
st.subheader("Forecast 🕒")
if st.session_state.ingested and st.session_state.ingested.tables:
    all_tables = list(st.session_state.ingested.tables.keys())
    tchoice = st.selectbox("Select table", all_tables)
    df = st.session_state.ingested.tables[tchoice]

    # Try construct Date from Year/Month if needed
    #if 'Month' in df.columns and 'Year' in df.columns and 'Date' not in df.columns:
     #   try:
      #      tmp = df.copy()
       #     tmp['Date'] = pd.to_datetime(tmp['Year'].astype(str) + '-' + tmp['Month'].astype(str) + '-01', errors='coerce')
        #    if tmp['Date'].notna().sum() > 0:
        #        df = tmp
         #       st.session_state.ingested.tables[tchoice] = df
        #except Exception:
         #   pass

    
    #num_cols = df.select_dtypes(include='number').columns.tolist()
    #dcol = st.selectbox("Date column", options=df.columns)
    #vcol = st.selectbox("Value column", options=num_cols or df.columns.tolist())
    #horizon = st.slider("Forecast horizon (periods)", 3, 36, 12)
    #season = st.number_input("Seasonal period (optional)", min_value=0, max_value=365, value=0)

    # date column selection + robust coercion
    dcol = st.selectbox("Date column", options=list(df.columns))
    if not pd.api.types.is_datetime64_any_dtype(df[dcol]):
        df[dcol] = parse_date_series(df[dcol])
    vcol = st.selectbox("Value column", options=list(df.select_dtypes(include='number').columns) or list(df.columns))

    model = st.selectbox("Model", ["ETS", "ARIMA", "RF"], index=0)
    horizon = st.slider("Horizon (periods)", 3, 36, 12)
    season = st.number_input("Seasonal period (optional)", min_value=0, max_value=365, value=0)

    if st.button("Run Forecast"):
        try:
            fc = Forecaster(df, dcol, vcol, season if season>0 else None)
            out, diag = fc.fit_predict(model=model, horizon=horizon)
            st.write("**Diagnostics**", diag)
            st.line_chart(out.set_index('Date')['forecast'])
            st.dataframe(out)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
else:
    st.info("Upload data and build the index to enable forecasting.")

st.caption("Tip: Compute KPIs via SQL (e.g., Total_GHG_Emissions by Plant & Month) then ask follow-ups in chat.")