# Semantic cache: Redis keys, values, domains, and hashes

## First: the names mean different things

| Term | What it means in this app | Is it a physical Redis box? |
| --- | --- | --- |
| Redis database | The parent Redis server/database that stores all records. | Yes. |
| `semantic_cache_v3` | The Redis search index over cache records. | An index, not a folder. |
| `cache:v3:` | Prefix applied to each cache record's Redis key. | No. |
| `domain` | A stable ID for the repository being searched. | No; it is a field on each record. |
| `coding_agent` | An agent namespace, stored in `agent_namespace`. | No; it is a field on each record. |

`coding_agent` and `medical` are therefore labels used to isolate records during
search. They are not separate Redis databases or actual nested boxes.

## One cached answer: exactly what is the key and what is the value?

For this example, assume:

```text
repository domain = repo_A_91ab...e2
agent namespace   = coding_agent
real LLM model    = openrouter/free
question          = How does billing work?
```

```text
Redis database
│
└── Redis KEY: cache:v3:7fa91c2d48e3b54091ad90bb3e7f812a
    │
    └── Redis VALUE: a hash/map with fields
        │
        ├── query_text   → "How does billing work?"
        ├── response     → "Billing works by assigning plans..."
        ├── domain       → "repo_A_91ab...e2"
        ├── agent_namespace → "coding_agent"
        ├── model        → "openrouter/free"
        ├── query_vector → [0.012, -0.183, 0.447, ...]
        ├── created_at   → 1788770000.12
        └── TTL          → 86401 seconds remaining/assigned
```

In short:

```text
KEY   = cache:v3:<hashed cache-entry identity>
VALUE = all fields below that key: question, answer, filters, vector, timestamp
```

The `response` is **not** the whole Redis value by itself. It is one field
inside the Redis hash/map that is the value.

## Two agent namespaces, one Redis database

Here is the mental model with a coding agent and a hypothetical medical agent.
They share one Redis database and one semantic-cache index, but every entry has
an agent/model label.

```text
Redis database
│
└── semantic_cache_v3 search index
    │
    ├── KEY: cache:v3:aa11...
    │   VALUE:
    │   ├── domain = repo_A_91ab...e2
    │   ├── agent_namespace = coding_agent
    │   ├── model  = openrouter/free
    │   ├── query_text = "How does billing work?"
    │   └── response   = "..."
    │
    └── KEY: cache:v3:bb22...
        VALUE:
        ├── domain = medical_44cd...90
        ├── agent_namespace = medical_agent
        ├── model  = openrouter/free
        ├── query_text = "What are common migraine triggers?"
        └── response   = "..."
```

When the coding agent searches, it asks Redis for the nearest semantic match
with both filters below:

```text
domain          == repo_A_91ab...e2
agent_namespace == coding_agent
model           == openrouter/free
```

So it cannot retrieve the medical entry. Conversely, a medical agent filters
for its own `domain` and `model` label.

Important terminology note: in the current code, `domain` means **repository
identity**, not agent type. `coding_agent` is part of `model`; a medical agent
would use a different namespace such as `medical_agent`.

## Why are hashes used?

The app uses two separate SHA-256 hashes. A hash is a deterministic function:
the same input always produces the same fixed-looking output.

### 1. Repository path → domain hash

```text
C:\projects\billing-service
        │
        ▼ SHA-256, shortened
repo_A_91ab...e2
```

This gives each repository a short, stable identifier. It avoids putting a
long, platform-specific path in every cache record and prevents two repositories
from sharing answers accidentally.

### 2. Scope + question → Redis entry-ID hash

```text
repo_A_91ab...e2
  + coding_agent
  + openrouter/free
  + How does billing work?
        │
        ▼ SHA-256, shortened
7fa91c2d48e3b54091ad90bb3e7f812a
        │
        ▼
Redis key: cache:v3:7fa91c2d48e3b54091ad90bb3e7f812a
```

This hash makes a compact, Redis-safe, deterministic key. It also means the
same question can coexist in different repository or agent/model scopes:

```text
repo A + coding_agent + same question → cache:v3:aa11...
repo B + coding_agent + same question → cache:v3:cc33...
repo A + medical_agent + same question → cache:v3:dd44...
```

In this app, “repo A” means the current working directory—the project root path used when the CLI starts.

The hash is **not** what makes the cache semantic. Semantic matching comes from
`query_vector`, the embedding of the question. The hash is only for naming the
individual Redis record.

Also, hashes here are for identity and compact keys, not secrecy: the original
question and answer are still stored in `query_text` and `response`.

## Lookup flow

```text
User asks: "Explain billing limits"
            │
            ▼
Create query_vector (an embedding; not a hash)
            │
            ▼
Search semantic_cache_v3 index
  where domain          = repo_A_91ab...e2
    and agent_namespace = coding_agent
    and model           = openrouter/free
            │
            ▼
Nearest matching record has enough similarity?
       │ yes                         │ no
       ▼                             ▼
Return its response             Call the agent, then store a new record
```
