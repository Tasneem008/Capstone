# SARP-Agent

A study-abroad counselling chat website for students, built on the SARP-Net continuation model.
Students chat in English, Banglish or Bengali; the agent answers their questions, builds their plan
(destination, intake, budget, result, research level), scores it with SARP-Net and focuses its advice
on what the model and the project's findings say matters most. Counsellors see the prediction, SHAP
drivers and evidence for every chat.

## Run it

1. Install the Python packages (Python 3.12, from the repository root):
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt -r agent/requirements.txt
   ```
2. Install [Ollama](https://ollama.com) and download the model: `ollama pull gemma3:4b`
3. Start the website from the repository root:
   ```
   .venv\Scripts\python.exe -m agent.server
   ```
   Students: http://127.0.0.1:8000. Counsellors: http://127.0.0.1:8000/counsellor

Without Ollama the site still works in offline mode: answers come from past counsellor conversations.

To put the site online (Vercel, a free Neon database, the PC's GPU as the AI with a free hosted backup),
follow [DEPLOY.md](DEPLOY.md).

## How a reply is made

| Part | File | Source of truth |
|---|---|---|
| Reading details from chat | `extract.py` | Rules for English, Banglish and Bengali, plus quick-reply buttons |
| Prediction and SHAP drivers | `risk_tool.py` | SARP-Net Gradient Boosting model, same code as the dashboard |
| Evidence for each driver | `findings.py` | `results_ablation/`, `experiments/results/e2_*`, `results_paper2/` |
| Answers to open questions | `chat.py`, `llm.py` | Any OpenAI-compatible model (Ollama locally; team PC + backup online), given real counsellor answers to similar questions (`retriever.py`) |
| PC model gateway | `llm_gateway.py`, `start_gateway.ps1` | Lets the online site use the PC's GPU safely |
| Fixed wording (questions, plan, refusals) | `prompts.py` | Written in all three languages |
| Storage | `store.py` | SQLite at `../agent_data/sarp_agent.db` on the PC; Postgres when `DATABASE_URL` is set |

Students never see their probability or risk tier; only counsellors do.

## Data

The counsellor-example search reads `../data/splits/train.jsonl` and `val.jsonl` (made by
`data_prep/parse_conversations.py`, kept outside the repository because the conversations are
confidential). The test split is never used, so it stays clean for evaluation.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SARP_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | AI endpoint (OpenAI-compatible) |
| `SARP_LLM_API_KEY` | empty | Key for that endpoint (the gateway key online) |
| `SARP_LLM_MODEL` | `gemma3:4b` | Model name; set to the fine-tuned model's name once imported |
| `SARP_LLM_FALLBACK_*` | not set | Optional backup AI (base URL, key, model) |
| `DATABASE_URL` | not set | Postgres connection; without it chats go to SQLite |
| `SARP_POLITICAL_PHASE` | `Newly Elected` | Phase used for every prediction |
| `SARP_BDT_PER_USD` | `122` | Exchange rate for budgets given in taka |
| `SARP_DATA_DIR` | `../data` | Folder holding `splits/` for counsellor examples |
| `SARP_DB_PATH` | `../agent_data/sarp_agent.db` | Chat database |
| `SARP_COUNSELLOR_PIN` | not set | If set, the counsellor page asks for this PIN |

## Tests

```
.venv\Scripts\python.exe -m unittest agent.test_risk_tool agent.test_chat agent.test_deploy -v
```
Set `TEST_DATABASE_URL` to a Postgres database to also run the storage tests against Postgres.
