# Semantic-cache Redis inspection example

Command run:

```powershell
poetry run python inspect_semantic_cache.py --all-versions
```

Observed cache entry:

```text
Redis key:          cache:v3:21152ef17c3093c6da99506d7ab0a068
TTL:                558 seconds
Domain:             9ad91afd35ce611e
Agent namespace:    coding_agent
LLM model:          openrouter/free
Created at:         2026-09-07T17:05:58+05:30
Question:           how billing works?
Response:           ## How Billing Works

**`billing.py`** is the entire billing module — it's plan-based **usage
gating**, not financial invoicing. Here's the breakdown:

### Plan Tiers
Three tiers defined in `PlanTier` enum, each with a hard cap on how many tasks
a user can create:

| Plan | Limit |
|---|---|
| `FREE... [truncated; use --full-response]

Embedding:          1536 bytes (binary float32 vector)
Original agent time:46911.6 ms
Original total time:47006.2 ms
```

## What this tells us

```text
Redis key
└── cache:v3:21152ef17c3093c6da99506d7ab0a068
    └── One semantic-cache record.

Domain
└── 9ad91afd35ce611e
    └── Hash representing the current repository root.

Agent namespace
└── coding_agent
    └── Identifies this application/agent role.

LLM model
└── openrouter/free
    └── The actual LLM-model scope for the response.

Question + embedding
└── The original question is stored as text and as a 384-dimension float32
    vector (384 × 4 bytes = 1536 bytes). Redis uses that vector to find
    semantically similar future questions.

Original timings
└── The original cache miss took about 47 seconds:
    agent work:       46,911.6 ms
    total before save:47,006.2 ms
```

When a new-session question is sufficiently similar and passes the repository,
agent-namespace, and model filters, a cache hit can return this stored response
without repeating the roughly 47-second agent execution.
