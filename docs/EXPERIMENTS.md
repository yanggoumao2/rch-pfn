# Dataset construction and research evaluation protocol

## Three levels of data

1. **Engineering fixture:** `configs/smoke.yaml`, 12 independent systems × 2 episodes,
   six observed frames each, procedural RGB. Runs exercise the entire gradient,
   storage, resume and evaluation path. They cannot establish physical performance.
2. **Development pilot:** `configs/pilot.yaml`, 30 systems × 3 episodes, eight frames,
   variable object count, command motion, hidden relation changes and occlusion. The
   procedural renderer remains a kinematic image generator.
3. **Research collection:** record the selected LabUtopia setup using the same sensor
   contract. Keep RGB originals, calibration, command/actual execution, exposure/arrival
   times, FK, object and part IDs, interventions and ground-truth poses. Import through
   the validated recorder API. The simulator is responsible for actual rigid-body
   dynamics and independent completion criteria; this package does not invent these.

Starter procedural prior: xyz span ±0.12 m, local orientation span ±0.6 rad,
change probability 0.6, occlusion probability 0.15, image noise standard deviation
0.015 in [0,1] RGB, sample interval 0.1 s. These are engineering defaults and must be
calibrated to the actual scene before physical research. See the generator for exact
sampling semantics. Do not report these numbers as measured instrument tolerances.

For physical collection, vary geometry/object instances, camera/background/light,
initial grasp, target pose, intervention time/amplitude and sensor delay separately.
Include normal commanded motion, no-event controls, persistent grasp changes,
target mechanism changes, combinations and unresolved ambiguity. Build the
train/validation/test allocation **before** drawing temporal prefixes.

| Partition | Proposed research collection | Purpose |
|---|---:|---|
| Train | 70% system groups | Fit weights and all normalizers |
| Validation | 15% groups | Select checkpoint/hyperparameters and fit calibration |
| Test | 15% untouched groups | Final estimates with group-level confidence intervals |
| Additional OOD | disjoint mechanisms, geometry and compositions | Separately report generalization |

The percentage is not sufficient if object/mesh families repeat across system IDs.
Group such shared factors under one split owner in the collector manifest. Reusing a
trajectory or an object-family realization across splits is leakage even if filenames
differ. Each scene/session must retain immutable source IDs and a provenance hash.

## Required comparisons

Train with identical task priors, sensor inputs, controller primitives, split groups,
optimization budgets and multiple seeds (start with 3, report individual values):

* Full shared two-pass model.
* One-pass core: `model.rounds: 1`; report its lower compute rather than pretending
  it is compute-matched.
* Two passes without hypothesis-conditioned reading: `model.condition_history: false`.
* A matched-budget temporal context model and explicit geometry/state-estimation
  baseline; do not claim the one-pass ablation is every possible Transformer baseline.
* Physical controller comparisons: retain-all, full-reset and selective update, using
  identical sensor/control budgets. Their physical implementations depend on the
  selected simulator/controller and are not fabricated by the RGB generator.

Do not change the data and architecture simultaneously and attribute everything to
PFN training. Pretrained versus random initialization is a separate factor.

## Measurements

The implemented evaluator reports held-out joint state NLL, regime time scores,
forecast NLL, MAP-mode coordinate RMSE and outcome Brier score. Negative continuous
NLL is possible; compare using the same chart and normalizer. Inactive label counts
are logged, including missing geometry supervision. Brier/NLL do not by themselves
certify a calibrated controller.

Evaluation uses a fixed one-scene forward and fixed predictive quadrature per scene,
then averages scene scores. Valid label/dimension counts are summed. This keeps
results invariant to the training batch size and padding composition; it is an explicit
scene-balanced estimator, not a globally label-weighted score. Validation and test use
this same rule. Masked labels do not contribute to the corresponding scene score.

Before a paper: add credible-region coverage at several levels, symmetry-aware
rotation error, identity association accuracy, sensitivity to context length, and
cross-seed/system confidence intervals. Physical rollout evaluation must separately
measure success/failure, erroneous continuation, recovery latency, force/collision
limits, observation and action budgets, total elapsed time including failures, and
abstention. Never relabel a dry-run acceptance as successful robot execution.

History-mechanism tests should compare pairs with nearly identical current images
but different informative histories, changed versus unchanged relations, irrelevant
old evidence and misleading first-round hypotheses. Positive evidence requires
better calibrated predictions or lower recovery cost, not merely nonzero attention.

## Reproduction and publication

Retain resolved configs, source hashes, dataset manifest hashes, environment lock,
seed, checkpoint and all validation/test outputs. Select using validation only. Keep
test results even if a simple baseline wins. Do not tune on the held-out test report.
Large data/checkpoints belong outside Git and need explicit redistribution rights.
Fill author order, license, citation and venue metadata only when actually decided.
