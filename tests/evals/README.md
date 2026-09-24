# Deterministic retrieval evaluation

This suite makes no judge-LLM calls and requires no extra packages. It matches
the expected file and symbol evidence from `golden_dataset.json` against the
actual top 80 retrieved chunks and reranked top 5 chunks.

Before a clean run, stop the API and delete any older chunks stored under the
evaluation user:

```powershell
poetry run python tests/evals/reset_eval_index.py --yes
```

This deletes only points whose `metadata.user_id` equals the configured
evaluation UUID. It does not delete the collection or other users' points.

Start the API with the fixture as its working directory:

```powershell
$appPython = poetry env info --executable
Push-Location tests/evals/fixtures/sample_project
& $appPython -m uvicorn ia_claude.api:app --port 8000
```

From another terminal at the repository root, run:

```powershell
python tests/evals/run_retrieval_eval.py
```

Per-question results are written to `tests/evals/results/retrieval.json`, and
aggregate means are written to `tests/evals/results/summary.json`. Metrics that
are `null` because a case has no positive expected evidence are excluded from
the means.
