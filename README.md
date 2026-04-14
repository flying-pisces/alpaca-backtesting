# alpaca-backtesting

Dashboard and backtesting harness for running multiple trading algorithms across
multiple Alpaca paper accounts.

## What this is

A personal control panel for the 5 SignalPro Pulse algorithms
(**Degen / Surge / Moderate / Sentinel / Fortress**) running against Alpaca's
paper trading API. Alpaca caps paper accounts at ~3 per user, so algorithms are
grouped by risk/DTE into two accounts:

| Bucket | Account | Algos | Rationale |
|---|---|---|---|
| **A — Aggressive** | `account_a` | Degen, Surge, Moderate | Short DTE (0–45d), higher turnover |
| **B — Conservative** | `account_b` | Sentinel, Fortress | Long DTE (45–365d), capital preservation |

Per-algo P&L attribution inside a shared account uses a `client_order_id`
prefix convention (e.g. `degen_<uuid>`).

## Layout

```
alpaca-backtesting/
├── config/
│   ├── algos.yaml           # 5-algo registry — add new algos here
│   └── accounts.yaml        # account → algos mapping
├── src/alpaca_dashboard/
│   ├── settings.py          # loads config + .env
│   ├── alpaca_client.py     # multi-account wrapper around alpaca-py
│   └── app.py               # Streamlit dashboard
├── data/                    # SQLite/backtest artifacts (gitignored)
├── scripts/run_dashboard.sh
├── requirements.txt
└── .env.example             # copy to .env and fill in
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste paper keys for account A and account B
```

## Run

```bash
./scripts/run_dashboard.sh
# opens http://localhost:8501
```

Pages: **Overview** (equity per account, algo mapping), **Accounts**
(positions per account), **Algos** (per-algo config + order count),
**Orders** (filterable order history with algo attribution).

## Adding a new algo

1. Append an entry to `config/algos.yaml`.
2. Add its `id` to the relevant account's `algos:` list in `config/accounts.yaml`.
3. When your strategy submits orders, set `client_order_id` to start with
   `<algo_id>_` — the dashboard will attribute fills automatically.

## Sibling project

Strategy logic lives in `market_pulse/` (signal generation, scanners, PGI).
This repo is the *management layer*: credentials, account routing, monitoring,
and historical record collection.
