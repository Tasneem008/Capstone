# Put SARP-Agent online (Vercel)

The website runs on **Vercel** (free Hobby plan), so it opens on any phone or laptop. Three free services
sit behind it:

| Part | Where it runs | Why |
|---|---|---|
| Website, chat engine, SARP-Net model | Vercel, Mumbai region (`bom1`) | Closest region to Dhaka |
| Chats, call-backs, feedback, counsellor examples | Neon Postgres (free, added from inside Vercel) | Vercel has no permanent disk |
| AI replies (primary) | The team PC's RTX 3060, through a secure gateway and an ngrok link | Free, private, runs the fine-tuned model later |
| AI replies (backup) | A free hosted open-model API (Groq or Google AI Studio) | Answers when the PC is off |

If both AI options are down, students still get answers drawn from past counsellor conversations.

## One-time setup

### 1. Vercel project
1. Sign up at vercel.com (Hobby plan) with GitHub.
2. **Add New, Project**, import `Tasneem008/Capstone`. The repository owner may need to approve
   Vercel's access to the repo.
3. Keep **Root Directory** as the repository root. Vercel reads `pyproject.toml` and `vercel.json`.
4. Under **Settings, Git**, choose the `Extended` branch as the production branch (or merge into `main`).

### 2. Database
1. In the project, open **Storage, Create Database, Neon (Postgres)**, free plan, region
   **Singapore** (nearest to Mumbai).
2. Connect it to the project. Vercel adds `DATABASE_URL` for you.

### 3. Backup AI (free)
Pick one and create a free API key:

| Provider | `SARP_LLM_FALLBACK_BASE_URL` | Example `SARP_LLM_FALLBACK_MODEL` |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Google AI Studio | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemma-3-27b-it` |

Check the provider's current model list and free limits first: they change often. Put the key in
`SARP_LLM_FALLBACK_API_KEY`.

### 4. The team PC as the primary AI
1. Install ngrok (`winget install ngrok.ngrok`), sign up for free, and run `ngrok config add-authtoken <token>`.
2. In the ngrok dashboard, claim your free **static domain** (for example `sarp-agent.ngrok-free.app`).
3. Make a long random gateway key and keep it secret:
   `.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"`
4. Start Ollama, then the gateway and the tunnel:
   ```
   $env:SARP_GATEWAY_KEY = "<the key>"
   .\agent\start_gateway.ps1 -Domain sarp-agent.ngrok-free.app
   ```
   Only the gateway is exposed, and it rejects any request without the key. Ollama stays private.

### 5. Environment variables in Vercel
**Settings, Environment Variables** (Production and Preview):

| Name | Value |
|---|---|
| `SARP_LLM_BASE_URL` | `https://<your-static-domain>/v1` |
| `SARP_LLM_API_KEY` | the gateway key from step 4 |
| `SARP_LLM_MODEL` | `gemma3:4b` (later: the fine-tuned model's Ollama name) |
| `SARP_LLM_FALLBACK_BASE_URL`, `_API_KEY`, `_MODEL` | from step 3 |
| `SARP_COUNSELLOR_PIN` | a PIN for the counsellor page (required: the page stays off without it) |
| `SARP_POLITICAL_PHASE` | `Newly Elected` (or the current phase) |
| `SARP_BDT_PER_USD` | `122` (update to today's rate) |
| `DATABASE_URL` | set automatically by Neon |

### 6. Upload the counsellor examples (once)
The conversations are confidential and stay out of GitHub. Upload the example exchanges straight to the
database from the PC that has the `data` folder (copy `DATABASE_URL` from **Storage, Neon, .env.local**):
```
$env:DATABASE_URL = "<Neon connection string>"
.venv\Scripts\python.exe -m agent.seed_examples
```

### 7. Deploy
Push the branch. Vercel builds a preview; open it and check:
- the build log installs from `pyproject.toml` (not `requirements.txt`);
- a chat streams a reply, the plan card fills in, and **/counsellor** asks for the PIN.

Then **Promote to Production**. Set `VERCEL_ANALYZE_BUILD_OUTPUT=1` once to see the bundle size.

## Day to day
- **Before a demo:** start Ollama and `start_gateway.ps1` on the PC. The counsellor page header shows
  whether "team PC" and "backup" are up.
- **Switching to the fine-tuned model:** import it into Ollama on the PC
  (`ollama create sarp-agent -f Modelfile`), then set `SARP_LLM_MODEL=sarp-agent` in Vercel and
  redeploy. Nothing else changes.
- **Message limits:** 20 per 10 minutes and 200 per day per browser, and 60 / 600 per network address.
  Change them with `SARP_RATE_BROWSER_10MIN`, `SARP_RATE_BROWSER_DAY`, `SARP_RATE_IP_10MIN` and
  `SARP_RATE_IP_DAY` (0 turns one off).
- **First message after idle** can take 5-15 seconds while Vercel loads the model and libraries.
  Opening the page already starts this warm-up.

## Privacy
- Student chats and call-back numbers are stored in Neon. Students see a note about this on the welcome
  screen.
- The backup AI provider receives messages only while the team PC is off.
- Never commit `data/`, `agent_data/`, `.env` files or keys. `.gitignore` covers them, and
  `.vercelignore` uploads only the files the site needs.
