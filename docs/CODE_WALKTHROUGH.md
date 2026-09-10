# LinkMind AI: interview code walkthrough

## 30-second introduction

"Maine LinkMind AI banaya hai, jisme user ek public webpage ka URL deta hai
aur us page ke content par questions pooch sakta hai. Backend FastAPI mein hai.
App page ka text extract karta hai, usko chunks mein divide karta hai, relevant
chunks search karta hai, aur gpt-4o-mini se context ke basis par answer banata
hai. Hindi, English aur Hinglish queries support hoti hain. Deployment Docker
ke through Hugging Face Spaces par hai."

## Code kis order mein dikhana hai

| File | Responsibility | Important functions / objects |
| --- | --- | --- |
| `src/main.py` | API routes aur browser session state | `app`, `ingest_url`, `chat`, `SESSION_STATES` |
| `src/schemas.py` | Incoming JSON ki validation | `URLRequest`, `ChatRequest` |
| `src/ingestion.py` | Link se readable text lana aur chunks banana | `fetch_url_document`, `html_to_text`, `load_and_index_urls` |
| `src/retrieval.py` | Relevant chunks search karna aur answer banana | `LocalRetrievalQA`, `retrieve`, `invoke` |
| `src/config.py` | API key, model, timeouts, project paths | `OPENAI_API_KEY`, `LLM_MODEL`, `BASE_DIR` |
| `static/script.js` | Browser se ingest/chat requests aur session ID | `processUrl`, `handleChat`, `getSessionId` |
| `templates/index.html`, `static/style.css` | UI markup aur styling | URL input aur chat interface |
| `app.py` | Existing deployment command ko support karna | `from src.main import app` |

## 1. URL se data kaise aata hai?

1. `static/script.js` ka `processUrl()` URL aur `session_id` ko
   `POST /api/ingest` par bhejta hai. Browser ka Microlink call preview card ke
   liye hai; backend separately Q&A ke liye content fetch karta hai.
2. `src/schemas.py` ka `URLRequest` HTTP/HTTPS URL validate karta hai.
3. `src/main.py` ka `ingest_url()` API key check karke
   `src/ingestion.py` ka `load_and_index_urls()` call karta hai.
4. `fetch_url_document()` requests se page download karta hai.
   HTML/XML response ke liye `html_to_text()` scripts, styles aur kuch other
   non-content tags remove karke readable text nikalta hai.
5. Direct fetch par HTTP 401, 403 ya 429 aaye toh Jina Reader try hota hai.
   Jina fail ho toh Microlink title/description jaisi metadata deta hai.
   Other HTTP/connection errors par existing error response milta hai.
6. 50 characters se zyada content ko `RecursiveCharacterTextSplitter`
   1000-character chunks mein split karta hai, with 150-character overlap.
   Overlap adjacent chunks ke beech context retain karne mein help karta hai.
7. `setup_qa_chain()` chunks se `LocalRetrievalQA` banata hai. Route is chain,
   chunks aur active URL ko session ke liye memory mein store karta hai.

## 2. Search aur answer kaise banta hai?

1. Browser ka `handleChat()` question aur wahi `session_id` ko
   `POST /api/chat` par bhejta hai.
2. `src/main.py` ka `chat()` matching session ki QA chain leta hai aur
   `invoke({"input": question})` call karta hai.
3. `src/retrieval.py` ka `expand_query()` LLM se Hindi/English keywords leta
   hai. Isse Hinglish question aur Hindi content match karne mein help milti
   hai. Expansion fail ho toh original question use hota hai.
4. `retrieve()` har chunk ko score deta hai:
   common words ki count + common character trigrams ki count / 8
   + expanded query ke exact phrase occurrences * 3.
   Highest positive scores wale up to 8 chunks select hote hain. Match na
   mile toh first 8 chunks use hote hain.
5. `invoke()` selected chunks ko context bana kar `SYSTEM_PROMPT` aur
   question ke saath `gpt-4o-mini` ko bhejta hai. Prompt model ko provided
   context se answer karne ko kehta hai; yeh correctness ki guarantee nahi hai.
6. Supported OpenAI API errors par `extractive_fallback()` relevant sentences
   select karke local answer deta hai. API key abhi bhi ingestion ke liye
   required hai; fallback ka matlab key-free mode nahi hai.

## Important interview distinctions

- Yeh ingested webpage ke chunks mein search karta hai. General internet
  search engine ki tarah naye links discover nahi karta.
- Current retrieval keyword aur character n-gram based hai. Is implementation
  mein embeddings ya FAISS vector database use nahi ho raha.
- Session dictionary ka purana `vector_store` key actually document chunks
  rakhta hai. Compatibility ke liye naam retain kiya gaya hai.
- `session_id` pages ke contexts alag rakhta hai, lekin authentication nahi
  hai. `STATE_LOCK` in-process dictionary access coordinate karta hai.
- Session data RAM mein hai. Restart par clear ho jata hai aur multiple
  workers ke beech share nahi hota. Redis/database future improvement hai.
- Microlink fallback full article ki jagah sirf metadata la sakta hai.
- Routes `async` hain, lekin current fetch/LLM calls synchronous hain.
  Higher concurrency ke liye async clients ya worker execution future work hai.

## Deployment aur verification

`Dockerfile` ka existing command `uv run uvicorn app:app --host 0.0.0.0
--port 7860` hai. Root `app.py` ab `src/main.py` se wahi FastAPI object import
karta hai, isliye deployment entry point same hai. README ka Hugging Face
metadata aur CI/CD pipeline bhi same hai.

Local run: `uv run uvicorn app:app --reload`

Tests: `uv run pytest -q`

Tests mein external HTTP aur LLM responses mock hote hain, taaki API calls
ka cost na ho aur validation, extraction, retrieval, fallbacks aur session
flow repeatably verify ho sakein. Live external services ki availability
aur Docker build ko separately verify karna hota hai.
