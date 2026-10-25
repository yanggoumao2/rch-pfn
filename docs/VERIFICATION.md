# Verification record — 2026-10-25

All execution was on the existing HP workstation, in the project's isolated environment,
using CPU. Neither workstation GPU workload was interrupted. No model training or test
was executed on the local RTX 5090.

| Check | Result |
|---|---|
| Pytest suite | 39 tests passed |
| Ruff correctness/import checks | Passed |
| Full default forward | 33,506,167 parameters; 8 joint modes × 32 visual nodes × 6 coordinates; 1122 directed edges; finite outputs |
| Gradient path | Nonzero finite final-state → first-state gradient; visual and history-conditioner gradients finite |
| Probability | Masked low-rank marginal matches dense MVN; scene mixture normalization; symmetry and missing-label cases |
| Causality | Reject unarrived exposures and duplicate sources; future actions cannot alter current state |
| Masks | Invalid pixels/labels cannot alter model evidence, assignment or loss; padded history excluded from streaming |
| Data | Deterministic bytes/hashes; system-disjoint splits; recorder round-trip; optional mixed-label batches |
| Resume | Two continuous steps and one step + restored step produce exactly identical parameters |
| Evaluation | Scores and total counts unchanged by configured batch grouping |
| Control | Bind prediction to exact command/start state/version; reject changes, expiry, missing identity/scale/calibration and blocked paths |
| Packaging | Python wheel and source distribution built |

The engineering pilot uses 30 independent synthetic systems × 3 episodes = 90 records:
63 train, 12 validation and 15 test records. It runs 100 optimizer steps on the
compact pilot configuration. This is a **procedural kinematic RGB fixture**, not a
physical contact dataset and not a trained robot recovery result. The authoritative
latest outputs are `runs/pilot-v2/` on the workstation; small reports are copied to
`reports/` in this project. Earlier smoke and pilot-v1 runs remain as development history.

Independent review found and prompted fixes to input/target dimension and dtype
integration, exact action binding, path checking, stream segment identity, crash-safe
locking, symmetry marginals, invalid-pixel matching, and batch-invariant evaluation.
Regression tests were added for those behaviors. A review is not proof that all future
configurations or physical integrations are error-free.

The independent final review closed every issue it had identified (no remaining
review P1/P2 in that scope). Root subsequently confirmed the final pilot completed.
The 15-record test diagnostic has translation-coordinate RMSE about **0.0370 m**;
this is not insertion-grade accuracy. A low Brier score on this small imbalanced
fixture is not a success-rate or calibration claim. Raw scores and fingerprints in
`reports/pilot-test.json` are retained so they can be interpreted without cherry-picking.

The final package does not claim calibrated uncertainty, learned causal discovery,
transparent-object perception, official TabPFN checkpoint reuse, or demonstrated
LabUtopia recovery gains. The current heads/priors and interfaces support subsequent
research, but actual scenario recordings, pretrained transfer, physical adapters,
strong baselines and multiple-seed experiments still require their own evidence.
