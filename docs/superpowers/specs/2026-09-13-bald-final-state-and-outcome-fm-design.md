# BALD arms for the final-state BNN and the outcome flow matcher

Adds three arms to the timeout-fix comparison so the paper's predictor-family
table covers two cells it currently misses: a BNN that regresses the final state
and acquires on BALD, and a scalar-outcome flow matcher that does the same.

Design date 2026-09-13. Campaign of record: `slurm_logs/timeout_fix/`, figures at
`/common/users/shared/pracsys/genMoPlan/docs/stochastic/timeout_fix/paper_draft/figures`.

## What already exists

Both predictors are built and tested. Nothing here re-implements them.

| piece | where | status |
|---|---|---|
| BNN final-state family | `trainers/final_state_trainer.py`, `predictors/final_state_handle.py` | shipped, Plan 2, 152 tests |
| `bnn_mfvi_reg`, `bnn_ensemble_reg` configs | `configs/adaptive_v2/predictor/` | shipped |
| scalar-outcome flow matcher | `model/outcome_flow_matcher.py`, `trainers/outcome_fm_trainer.py`, `probability/outcome_fm.py` | shipped; run on stochastic pendulum, seeds 42/43/44 |
| BALD acquisition | `strategy/decomposition.py`, `configs/.../acquisition/decomp_epi_bald.yaml` | shipped |

## The gap

`DecompositionAcquisitionStrategy` requires a probability backend exposing
`estimate_members()` and raises otherwise (`decomposition.py:53-60`). Neither
target arm has one.

**Final-state family.** Binds `/probability: endpoint_mc`, which has no member
axis. The deeper problem is that adding one naively would be wrong:
`FinalStateModelHandle.predict_endpoint` draws a fresh weight sample on *every
call* by documented contract (`final_state_handle.py:48-52`), so the K endpoint
draws that form p mix epistemic and aleatoric variation together. BALD computed
over those draws would score sampling noise, not model disagreement. The weight
draw has to be pinned across the K head draws, and the handle's contract makes
that structurally impossible from inside the handle.

**Outcome FM.** A single deterministic model. No posterior, no members, no
epistemic axis at all. It has only ever been run as
`+experiment=fm_outcome_baseline` (`acquisition.d2_ratio=0`, fixed dataset),
which needs no acquisition signal.

## Component 1: posterior-pinned endpoint MC backend

New `PosteriorEndpointMCProbabilityBackend` in
`adaptive_roa/adaptive_v2/probability/posterior_endpoint_mc.py`, subclassing
`EndpointMCProbabilityBackend`.

Subclassing rather than writing a sibling is the point: `estimate()`,
`sample_endpoints()`, `p_invalid` and the whole calibration and evaluation path
stay byte-identical to the existing reg arms, so the only thing the new arm
changes relative to `bnn_mfvi_reg` is that acquisition gains a surface it did not
have. Nothing downstream of the predictor moves.

```
estimate_members(states) -> [S, N]:
    embedded = system.embed_state_for_model(system.normalize_state(x))
    if hasattr(posterior, "forward_all_members"):
        params = posterior.forward_all_members(embedded)    # ensemble: [M,B,P], exact
    else:
        params = posterior.forward_samples(embedded, S)     # mfvi:     [S,B,P]
    for s in range(len(params)):
        hits = sum over K of (system.classify_attractor(head.sample(params[s]), r) == 1)
        out[s] = hits / K
```

`member_sample_size` returns K, so BALD's binomial debiasing applies exactly as
it does for `EnsembleEndpointMCProbabilityBackend`.

Taking the ensemble branch through `forward_all_members` also removes, on the
acquisition path only, the sample-with-replacement bias documented at
`final_state_handle.py:57-74`. That bias is a property of the handle's
one-draw-per-call contract; this backend never goes through the handle, so it can
enumerate. Evaluation still goes through the handle and still carries the bias.
That asymmetry is deliberate and must be stated wherever these arms are compared
against `bnn_ensemble_reg`'s older numbers.

**Guard.** A `DeterministicPosterior` yields S identical draws and therefore BALD
= 0 everywhere, silently. `bind_model` refuses a posterior with no spread, the
same way `_EnsembleBackendBase` refuses fewer than 2 members.

**Cost.** S=64 weight draws over 250,000 q3d candidates is the same order as what
`SampledPosteriorClassifierProbabilityBackend` already does for `bnn_mfvi_a1` at
that candidate count in this campaign, so the budget is precedented rather than
estimated. The K head draws on top are cheap arithmetic on already-computed
params.

## Component 2: outcome-FM deep ensemble

Five independently trained outcome flow matchers behind one handle.

`EnsembleOutcomeFMTrainer` and `EnsembleOutcomeFMHandle` in
`adaptive_roa/adaptive_v2/trainers/ensemble_outcome_fm_trainer.py`, modelled on
`clf_ensemble` rather than on `EnsembleFlowMatchingTrainer`: members train
**sequentially**, because this is a small MLP over a 1-D flow and `clf_ensemble`
already trains an MLP ensemble sequentially in this same campaign at this same
budget. The `mp.spawn` harness exists because endpoint FM members cost ~50 h
each; that does not apply here, and it carries known hazards (children reload
modules from disk each epoch; user-site numpy can hang spawn jobs). Decision to
revisit: time the first cartpole epoch and move to the parallel harness only if
the measurement says to.

`EnsembleOutcomeFMProbabilityBackend` subclasses `OutcomeFMProbabilityBackend`
and adds `estimate_members()` returning each member's readout. With
`readout: exact` there is no MC noise inside a member, so `member_sample_size`
is None and BALD is unbiased.

**The correctness trap this arm must not fall into.** Calibration and evaluation
reach p through `model(x) -> logits` (`full_roa.py:1301`), while acquisition
reaches it through the backend. The handle's `forward` must therefore return
`logit(mean_m p_m)` so the two agree. If they diverge, calibration optimises a
different quantity than evaluation reports, silently, with both numbers looking
reasonable. `OutcomeFMProbabilityBackend.bind_model` already refuses a
readout mismatch for exactly this reason; the ensemble subclass extends that
guard to cover the marginal, and a test asserts
`sigmoid(handle(x)) == backend.estimate(x).p_success` to tolerance.

## Component 3: configs

New files only. No existing config is edited, because other sessions have live
jobs reading this tree and a config edit is the kind of change that reaches a
running job.

| file | contents |
|---|---|
| `configs/adaptive_v2/probability/posterior_endpoint_mc.yaml` | new backend, `n_posterior_samples: 64`, `num_mc_samples: 10` |
| `configs/adaptive_v2/probability/ensemble_outcome_fm.yaml` | new backend, `readout: exact` |
| `configs/adaptive_v2/predictor/bnn_mfvi_reg_bald.yaml` | `bnn_mfvi_reg` + new probability group + per-system width |
| `configs/adaptive_v2/predictor/bnn_ens_reg_bald.yaml` | `bnn_ensemble_reg` + new probability group + per-system width |
| `configs/adaptive_v2/predictor/fm_outcome_bald.yaml` | ensemble trainer, `n_members: 5`, per-system width |

`predictor.name` stays `bnn_mfvi_reg` / `bnn_ensemble_reg` on the two BNN
configs. That field keys the export registry
(`probabilistic_classifier/registry.py`), and the predictor genuinely is
unchanged; only acquisition differs. This matches how `epi_bald_greedy` and
`dir00_s42` both carry `predictor.name: fm_ensemble`. The arm identity lives in
the run-directory name, not in `predictor.name`.

The outcome arm takes a new `predictor.name` of `fm_outcome_ens`, because it is a
different model class and would otherwise load the wrong checkpoint class if
exported. Note that `fm_outcome` is not registered for export today at all;
export stays out of scope here, unchanged from the shipped state.

**Naming, stated once so the three levels are not confused.** Config file name
equals arm name; `predictor.name` is a separate axis and names the *model*, not
the arm.

| config / arm name | `predictor.name` | run dir |
|---|---|---|
| `bnn_mfvi_reg_bald` | `bnn_mfvi_reg` | `tf_<sys>_<level>_bnn_mfvi_reg_bald` |
| `bnn_ens_reg_bald` | `bnn_ensemble_reg` | `tf_<sys>_<level>_bnn_ens_reg_bald` |
| `fm_outcome_bald` | `fm_outcome_ens` | `tf_<sys>_<level>_fm_outcome_bald` |

**Duplication risk and its guard.** The two BNN configs copy the `final_state`
block of their base arm. A test asserts the copied block matches the base arm
field-for-field *except* `hidden_dims`, so drift is caught rather than
discovered in a result.

## Network width

All three new arms use `${model_dims.mlp_hidden_dims}`
(`configs/adaptive_v2/model/system_dims/<system>.yaml`) rather than the
hardcoded `[256, 512, 256]` that `bnn_mfvi_reg` and `fm_outcome` carry today.
That hardcoded value is the pendulum entry, and on q3d it means a pendulum-sized
network against millions of training rows. This is the same defect
`bnn_mfvi_a1.yaml` was written to fix, and `bnn_mfvi_a1` is the arm these sit
beside in the table.

Stated cost, decided by the user on 2026-09-13: `fm_outcome_bald` is then no
longer capacity-matched to `classifier.yaml`, which was the original defining
contrast of the outcome-FM ablation. These arms are therefore not comparable to
the pendulum `fm_outcome` runs of that ablation, and the new arms differ in width
from `clf_ensemble` within this campaign. The table was already mixed on this
axis; per-system is the better-engineered side of it.

## Timeout intermediates

Every run carries `++data_source.timeout_intermediates=drop`, matching the rest
of the campaign (commit `f16314f`).

Verified rather than assumed to reach all three arms. The drop happens in
`get_all_endpoint_pairs_from_trajectory` (`adaptive/data_source.py:518`), and
`build_endpoint_dataset` (`:557`) returns starts, ends *and* labels from that one
builder. Both `save_endpoint_dataset` (`:623`) and `save_classification_dataset`
(`:670`) call it, so a timeout trajectory contributes only (x_0, x_T) to the BNN
final-state arms and only x_0 to `fm_outcome_bald`'s classification rows. One
override covers all three; no code change is needed.

The mode needs the collector horizon and raises without it (`:73`, `:308`), so a
missing horizon fails loudly rather than reverting to `keep`. Acceptance check:
the line `timeout_intermediates=drop: N of M trajectories time out` must appear
in each arm's epoch-0 log before the campaign runs past its first cell.

Expect this to move cartpole results substantially (~5% of trajectories, ~30% of
rows) and q2d/q3d barely (timeouts under 0.3%).

## Run matrix

Three arms, nine cells, seed 42. 27 runs.

| system | levels | initial + per-epoch | extra |
|---|---|---|---|
| cartpole | baseline, low, med | 500 + 150 | `controller=safe_explorer_ppo` |
| quad2d | baseline, smooth, loud | 2000 + 600 | `noise_family=corridor_sine_ambient` |
| quad3d | f_0.00, f_0.12_a0.03, f_0.40_a0.04 | 5000 + 1000 | `+controller=ppo_1500K`, `acquisition.n_candidates=250000` |

Copied verbatim from `slurm_logs/timeout_fix/manifest.tsv`:
`acquisition=decomp_epi_bald acquisition.d2_ratio=1.0 acquisition.selection_rule=greedy`,
`num_workers=0 adaptive_v2.filter_confident_pairs=false seed=42`,
`++data_source.timeout_intermediates=drop n_epochs=11`,
`predictor.lightning_trainer.enable_progress_bar=false`, and an explicit
`output_dir`.

Output root is the existing
`/common/users/shared/pracsys/adaptive_roa_experiments/timeout_fix/`, run dirs
named `tf_<sys>_<level>_<arm>`. Confirmed safe with the session that owns the
campaign: its launcher resolves rows from `manifest.tsv` by exact name and its
status scripts iterate manifest rows, so extra directories are inert and its
auto-resume cannot pick these up.

q3d is `ppo_1500K` on the 15k budget. The `ppo800k` in the figure filenames is a
naming defect (`scripts/paper/plot_timeout_fix.py:53` keys the system
`quad3d_ppo800k` while reading the ppo1500k CSV, and the key becomes the
filename), not a different controller. The real ppo_800k campaign in the scorer
is older pre-timeout-fix work and is not what these figures show. The 40k budget
is a separate campaign (`tf40_q3d_*`) and is out of scope.

## Scoring

Three names added to `ARMS` in `scripts/score_stoch_incremental.py` near line 59:
`bnn_mfvi_reg_bald`, `bnn_ens_reg_bald`, `fm_outcome_bald`. No new `CAMPAIGNS`
entry, since the root, prefix and ground truth are unchanged.

A name missing from `ARMS` is skipped **silently** and the scorer still exits 0,
so the acceptance check is that the three arms appear as rows in the level CSV
after the first scoring pass. Exit status is not evidence.

`predictor_of()` labels these `fm`, matching how the existing `bnn_*` arms are
labelled. The tag is a CSV label, not a branch: `stoch_prob_metrics.py` decides
continuous-versus-discrete from the number of distinct predicted values, not from
the tag. Worth noting that `fm_outcome_bald` under `readout: exact` emits a
continuous p_hat while the BNN arms emit K-quantised values, and the scorer
handles both, on its own, by that data-driven rule.

## No uniform controls

Decided by the user on 2026-09-13, against a reasonable objection from the
campaign-owning session that BALD-only arms confound predictor with acquisition.

The objection is correct in general and is noted here so nobody rediscovers it as
a surprise. It is not a change in practice: of the campaign's 44 runs, only the
FM family has both a BALD and a uniform arm; BNN-outcome and CLF are already
BALD-only, read against that single FM-uniform reference. These three arms adopt
the same structure. Adding matched `acquisition=direct d2_ratio=0` controls would
be another 27 runs and can be done later without invalidating these.

## Tests

Written before the implementation, per the repo's convention.

1. `estimate_members` returns `[S, N]` for an MFVI posterior and `[M, N]` for an
   ensemble, with every value in [0, 1].
2. The ensemble branch calls `forward_all_members`, not `forward_samples`. A
   posterior whose `forward_sample` is instrumented must see zero calls.
3. Weight pinning: with a stub head whose `sample` is the identity on its params,
   all K draws within one member are identical, and members differ. This is the
   test that would have caught scoring aleatoric noise as epistemic.
4. `member_sample_size == num_mc_samples` on the final-state backend, `None` on
   the outcome-FM ensemble backend.
5. `bind_model` raises on a deterministic posterior and on a 1-member ensemble.
6. Outcome-FM ensemble marginal agreement:
   `sigmoid(handle(x)) == backend.estimate(x).p_success` to tolerance.
7. Config drift: each `*_reg_bald.yaml` `final_state` block equals its base arm's
   field-for-field except `hidden_dims`.
8. End-to-end smoke: one adaptive epoch per new arm on cartpole at a tiny budget,
   asserting the acquisition diagnostics carry a non-zero `epistemic_mean`.

## Risks

**BALD may be near-zero on the MFVI final-state arm.** A mean-field posterior over
a regression head can be confidently wrong, and the K=10 binomial floor may swamp
the between-draw spread. Test 8's non-zero `epistemic_mean` assertion catches a
degenerate arm at smoke scale rather than after 27 runs. If it fires, raising K
is the first lever.

**Sequential 5-member training may be too slow on q3d.** Mitigated by measuring
the first cartpole epoch and keeping the parallel harness as a known fallback,
not by guessing now.

**These arms are not comparable to older `bnn_*_reg` or `fm_outcome` numbers**,
because of the width change and, for the ensemble arm, the enumeration change on
the acquisition path. They are comparable to each other and to the timeout-fix
campaign arms they sit beside, which is what the table needs.
