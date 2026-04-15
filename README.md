# alpaca-backtesting

Multi-account Alpaca paper-trading dashboard **and** walk-forward backtester
for the 5 SignalPro Pulse algorithms (Degen / Surge / Moderate / Sentinel / Fortress).

Strategy logic is imported from the sibling `market_pulse/` repo. This repo
is the *management layer* — credentials, per-algo routing, backtest harness,
coefficient tuning, and a Streamlit dashboard.

## Accounts

One algo per paper account (Alpaca caps paper accounts at ~3 per login,
so the 5 accounts span 2 logins):

| Algo | Account | Risk | DTE | Login |
|------|---------|------|-----|-------|
| Degen    | `degen`    | highest | 0–7   | 1 |
| Surge    | `surge`    | high    | 7–21  | 1 |
| Moderate | `moderate` | balanced| 21–45 | 2 |
| Sentinel | `sentinel` | low     | 45–90 | 2 |
| Fortress | `fortress` | lowest  | 90–365| 2 |

Secrets live in `.env` (gitignored) — see `.env.example` for schema.

## Layout

```
alpaca-backtesting/
├── streamlit_app.py             # Streamlit Cloud entrypoint (root)
├── pages/
│   ├── 1_Dashboard.py           # /Dashboard — backtest results
│   └── 2_Admin.py               # /Admin — run/stop + coefficient tuning
├── config/
│   ├── algos.yaml               # 5-algo registry
│   └── accounts.yaml            # one account per algo
├── src/alpaca_dashboard/
│   ├── app.py                   # home page module (imported by streamlit_app.py)
│   ├── strategies/              # copied from market_pulse (option builders)
│   ├── backtest.py              # walk-forward engine
│   ├── historical_data.py       # Alpaca bars adapter
│   ├── jobs.py                  # in-process thread registry
│   ├── store.py                 # SQLite (local) or Turso (remote)
│   ├── alpaca_client.py         # multi-account TradingClient wrapper
│   └── settings.py              # config + .env loader
├── data/                        # local SQLite (gitignored)
├── .streamlit/
│   └── secrets.toml.example     # template for Streamlit Cloud secrets
├── scripts/run_dashboard.sh
├── requirements.txt
└── .env.example
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste the 5 paper key/secret pairs
```

## Run

```bash
./scripts/run_dashboard.sh
# opens http://localhost:8501
```

- **/** — live paper accounts (equity, cash, positions)
- **/Dashboard** — backtest results: per-algo cards, cumulative P&L, drawdown,
  strategy breakdown, win-rate heatmap, filterable trade log, CSV export
- **/Admin** — per-algo controls: ▶ Run N batches · ⏹ Stop · tune
  `target_dte` / `pgi_entry` / `size_mult` sliders

## Deploy to Streamlit Cloud

Get a public URL (`https://<app>.streamlit.app`) that auto-deploys on every
push to `main`. Takes ~10 min end-to-end.

### 1. Create a Turso database (free, persistent)

Streamlit Cloud's filesystem is ephemeral — containers reset and wipe local
SQLite. [Turso](https://turso.tech) gives you SQLite-over-HTTP with a 9 GB
free tier. Install the CLI (`brew install tursodatabase/tap/turso`), then:

```bash
turso auth signup                                   # one-time
turso db create alpaca-backtesting
turso db show alpaca-backtesting --url              # → libsql://...
turso db tokens create alpaca-backtesting           # → long JWT string
```

Keep the URL and the token — you will paste both into Streamlit Cloud.

### 2. Push the repo to GitHub (public)

```bash
git remote add origin git@github.com:<you>/alpaca-backtesting.git
git push -u origin main
```

The repo is already safe to publish: `.env`, `.streamlit/secrets.toml`, and
`data/*.db` are all gitignored.

### 3. Create the Streamlit Cloud app

- Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
- Repo `<you>/alpaca-backtesting`, branch `main`, main file `streamlit_app.py`
- Under **Advanced → Secrets**, paste the contents of
  `.streamlit/secrets.toml.example` with real values filled in:
  - 5 × `ALPACA_<ALGO>_KEY` + `_SECRET`
  - `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN`
- Click **Deploy**.

Streamlit reads those secrets as environment variables, so the same code path
works locally (via `.env`) and on Cloud (via the secrets UI).

### 4. Push updates

Any commit to `main` redeploys automatically. Data written by `/Admin` runs
persists in Turso across restarts.

### Known Cloud-specific caveats

- **Long backtests may outlive the session.** Community Cloud keeps the server
  process alive across browser tabs, but a very long run can be killed if the
  container spins down. Start with `target_pulses = 50–120` and add more on
  demand.
- **Private repo requires a paid plan** (Streamlit Teams, $20/mo) — or
  self-host on Fly.io / Render instead.

## How backtests work

Per `(ticker, day, algo)`:

1. Compute PGI from 60-day price momentum + RSI (simplified — no news).
2. Compute 20-day annualised HV → sigma.
3. Ask `select_strategy_for_tier(S, sigma, expiry, dte, pgi, tier)` for a
   strategy (long call / iron condor / bull put spread / …).
4. Walk forward `dte` bars, score win/loss/neutral based on the strategy's
   payoff at expiry.
5. Persist to SQLite, tagged with a `job_id`.

Coefficients in `coefficients` table override the defaults per algo.
