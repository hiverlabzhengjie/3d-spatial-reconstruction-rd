# Stage Handoffs

Create one handoff document when each roadmap stage is closed:

```text
S00_HANDOFF.md
S01_HANDOFF.md
...
S07_HANDOFF.md
```

Do not create a completion handoff before the stage has actually met its
completion gate. Start from `HANDOFF_TEMPLATE.md`.

The latest completed handoff is required reading for the next stage. It should
record both software results and physical-world observations so that a fresh
Codex task can continue without relying on another task's conversation history.

After the completion gate and handoff are complete, create a dedicated
descriptive stage-close commit such as:

```text
stage(S03): complete person and backpack perception
```

Optionally add an annotated lowercase tag such as
`stage-03-person-backpack-perception`. Record the commit hash and tag in the
handoff. Do not create either before the stage is genuinely complete.
