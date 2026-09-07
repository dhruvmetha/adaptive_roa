---
type: experiment
status: idea
created: {{date}}
commit:
splits: []
metric:
systems: []
predictor:
thread:
verdict:
tags: [experiment]
---
# {{title}}

## Hypothesis
_(you)_ The falsifiable claim — what we're testing and expected direction.

## Plan
_(Claude)_ Code / data / config + the exact run command.

## Run
_(Claude, auto)_ job id · commit · config · date. Cross-ref research_journal/job_registry.tsv.

## Result + Verdict
_(Claude, auto)_ Numbers, split by every axis in `splits:`. Accept/reject on numbers only.
Cross-predictor verdicts MUST state abstention/coverage (F1 is not comparable across
methods with different abstention rates).

## Next
What this implies; the follow-up experiment.

## Discussion
_(you ↔ Claude — ask here; answered inline, dated `**[who YYYY-MM-DD]**`, newest at bottom.)_
