# Refund timeout prototypes (see the two Python files)

These are small prototypes of the video's timeout scenario. Both make real
OpenRouter model calls, but the payment provider and money are simulated.

The fixed graph is:

```text
START
  -> real LLM classifies intent (no tools available)
  -> plain code checks policy using trusted order data
  -> deterministic router
       |-- approved -> code calls provider with idempotency key -> END
       `-- denied ---------------------> END
```

The important point is not the refund rules. It is where authority lives:

- The customer message is untrusted.
- The intent-classification model receives no tools, so it cannot issue a refund.
- The order amount comes from the trusted order store, not the message.
- Plain Python routing chooses the allowed next graph node.
- The refund node is reachable only after `approved` becomes exactly `True`.
- Plain code owns the provider call and timeout retry.
- SQLite records that an attempt started before the provider call.
- The same `refund_id` is reused as the idempotency key on retry.

Set `OPENROUTER_API_KEY` in `.env`, then run it from the repository root:

```powershell
poetry run python prototypes/refund_state_machine/refund_workflow.py
```

The provider processes the first $100 refund but deliberately returns TIMEOUT.
Code retries with the same key, so two calls still move only $100 of fake money.

This demonstrates a state machine, not a production payment design. A real
system would also need transactional storage, authorization, idempotency keys,
auditing, concurrency controls, and possibly human approval.

## Unsafe comparison

`unsafe_refund_agent.py` gives a real LLM `issue_refund` on every model call.
The provider processes the first $100 refund but returns TIMEOUT. The LLM is
instructed to retry, so another call can pay another $100 because no idempotency
key exists.

Run the unsafe comparison with:

```powershell
poetry run python prototypes/refund_state_machine/unsafe_refund_agent.py
```

Because the unsafe decision comes from a real probabilistic model, its exact
behavior can vary. That variability is part of the problem being demonstrated.
