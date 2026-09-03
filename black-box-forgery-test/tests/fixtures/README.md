# Safe-agent fixtures

`page_clean.html` and `public.txt` are inert local fixtures. Each episode
creates its own `.env` with a unique deterministic synthetic canary; no real
credentials are stored in this repository. The mock-upload tool records
attempts in memory and never opens a network connection.
