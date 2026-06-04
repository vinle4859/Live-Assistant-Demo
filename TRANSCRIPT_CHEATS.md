# Transcript Cheats Guide

## Purpose
Transcript cheats are deterministic correction rules applied to STT output before routing. Use them for repeated, domain-specific STT mixups such as `remix` when the user actually said `Greenwich`.

The correction layer is intentionally guarded:
- It matches full words/phrases only.
- It is case-insensitive.
- It supports context terms so risky words are rewritten only in the right domain.

## Configuration
Set rules in `.env` using `VOICE_LOOP_TRANSCRIPT_CHEATS`.

Format:
- `wrong phrase=correct phrase`
- `wrong phrase=correct phrase|context1,context2`

Multiple rules are separated by semicolons.

Example:

```env
VOICE_LOOP_TRANSCRIPT_CHEATS=remix=greenwich|university,vietnam,tuition,hoc phi;green witch=greenwich|university,vietnam
```

## How Context Guarding Works
For a rule with context terms:
- At least one listed context term must appear in the same transcript.
- If no context term is present, no rewrite is performed.

This protects common real words. For example:
- `play my remix playlist` -> remains unchanged
- `how much is remix university tuition` -> `remix` can be corrected to `greenwich`

## Operator Playbook for Adding Rules
1. Add one rule at a time in `.env`.
2. Prefer context-guarded rules for common words (`remix`, `apple`, `metro`, etc.).
3. Keep `wrong phrase` as short as possible but specific.
4. Add 2-6 context terms that indicate the intended domain.
5. Run tests: `python -m pytest -q`.
6. Run a quick live check with 3 utterances:
   - Intended correction should trigger.
   - Unrelated phrase should not trigger.
   - Neutral phrase should remain unchanged.

## Rule Quality Tips
- Avoid global single-word rewrites without context.
- Avoid overlapping rules that map one wrong phrase to multiple targets.
- If a rule causes false positives, add or tighten context terms.

## Logging
When one or more rules apply, the pipeline logs the applied rule labels and the corrected transcript to make behavior auditable.
