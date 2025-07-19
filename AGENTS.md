# AGENTS.MD — Coding-Agent Playbook

This document is **the single source of truth** for any autonomous or semi-autonomous coding agent ("Agent") working in this repository.
It describes the workflow, guard-rails, and inline conventions that enable repeatable, high-quality contributions **for _any_ spec** located under `.kiro/specs/`.

---

## 1. Core Principles

1. **Task-Driven Development** – _Every_ change MUST be tied to a checklist item in a spec `tasks.md` file. No orphan code.
2. **Plan → Implement** – For complex tasks an Agent writes a plan first (see § 3). Code only after the plan is accepted/locked.
3. **Self-Verification Loop** – Run compiler / tests locally as often as needed; update test summary (§ 5) on every push.
4. **Greppable Inline Memory** – Use `AICODE-*:` comments to leave breadcrumbs for other Agents (§ 4).
5. **Small, Safe, Reversible Commits** – Prefer many focused commits over one massive diff.

---

## 2. Task Execution Protocol

> A human may trigger an Agent by saying e.g. `do task specs/whisperx-fastapi-server/10`.

1. **Locate the task**
   * Path pattern: `.kiro/specs/<spec-folder>/tasks.md`.
   * Task lines follow GitHub-style checkboxes `- [ ]` / `- [x]`.
2. **Analyse** the unchecked item: dependencies, affected code, tests, docs.
3. **If task requires > 3 non-trivial edits** – enter **Plan Mode** (§ 3). Otherwise jump to **Implement Mode** (§ 4 & § 5).
4. **After implementation**
   * Run full test suite.
   * Update the spec’s `tasks.md` – mark item `[x]` and append a _one-line_ changelog bullet below the task.
   * Commit with message `<spec>.<id> <short description>` (example: `whisperx-fastapi-server.10 add health endpoint`).

---

## 3. Plan Mode

Plans live in `plans/` and are named `###-objective-description.md` (increment `###`).
A plan MUST include:
* **Objective** – the task text verbatim.
* **Proposed Steps** – numbered, short, actionable.
* **Risks / Open Questions** – bullet list.
* **Rollback Strategy** – how to revert if needed.

Wait for human approval before touching code. After approval, reference the plan file at the top of the first commit message: `Plan: 012-health-endpoint`.

---

## 4. Inline Memory – `AICODE-*:` Anchors

Use language-appropriate comment tokens (`#`, `//`, `--`, etc.).

* `AICODE-NOTE:` – important rationale linking new to legacy code.
* `AICODE-TODO:` – known follow-ups not in current scope.
* `AICODE-QUESTION:` – uncertainty that needs human review.

Example (Python):
```python
# AICODE-NOTE: re-uses buffer sizing from legacy reader (see reader.py:42)
```
These anchors are **mandatory** when:
* Code is non-obvious.
* Logic mirrors or patches hard-to-find legacy parts.
* A bug-prone area is touched.

They are discoverable via `grep "AICODE-" -R`.

---

## 5. Test Summary Automation

Each spec keeps a lightweight test ledger `test-summary.md` next to its `tasks.md`.
After **every test run** an Agent MUST overwrite that file with the exact format:
```
# Test Summary
Passed: <NUM_PASSED>; failed: <NUM_FAILED>
## Passed
- <module.TestClass.test_name>
...
## Failed
### <module.TestClass.test_name>

Error: <first line of assertion error>
```
Full stack traces go inside a fenced block directly after each failed test.
This enables other Agents to quickly pick "first failing test", "second failing test", etc.

---

## 6. Self-Verification Checklist (before marking task done)

* [ ] `pytest -q` (or framework-equivalent) passes — or failures are *expected* and recorded in `test-summary.md`.
* [ ] Linter & formatter (`ruff`, `black`, `isort`) pass.
* [ ] No TODO left in scope unless explicitly out-of-scope.

---

## 7. Steering Updates

Agents MAY update `.kiro/steering/*.md` files when:
* A new technology or convention is introduced.
* An existing rule is changed.

Changes MUST be stated in the commit body: `Update steering: <file>`.

---

## 8. Fallback Behaviour

If uncertain, the Agent should:
1. Add an `AICODE-QUESTION:` inline comment.
2. Push the work behind a feature flag or in a draft PR.

---

*Happy coding! Remember: think deep, code sharp.*
