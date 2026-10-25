# RCH-PFN

**Relation-conditioned history inference from visual interaction contexts.**

Research implementation of a shared, two-pass visual context model. It learns a
joint distribution over entity states, uses each first-pass hypothesis to condition
how the same historical evidence is reread, and predicts outcomes of candidate
actions. There is no fixed `rotor/grasp/none` fault classifier and no embedded 10° answer.

**Status:** implementation and engineering validation; not a validated robot policy or
publication result. See [verification](docs/VERIFICATION.md) for measured checks and
[design correspondence](docs/DESIGN.md) for boundaries relative to the original design.

中文入口：[项目说明](docs/项目说明.md) · [数据集](docs/DATASET.md) ·
[架构和输入输出](docs/ARCHITECTURE.md) · [实验协议](docs/EXPERIMENTS.md) ·
[控制接口](docs/CONTROL.md)

## Quick start

Run these commands **on the workstation**. The local source mirror is not a training
location. Create an isolated Python 3.11+ environment and install the correct PyTorch
build for the chosen device, then:

```bash
python -m pip install -e '.[dev]'
rch-pfn generate --config configs/smoke.yaml --output data/smoke
rch-pfn validate-data --data data/smoke
rch-pfn train --config configs/smoke.yaml --data data/smoke --run runs/smoke --device cpu
rch-pfn evaluate --checkpoint runs/smoke/best.pt --data data/smoke --split test --output runs/smoke/test.json
rch-pfn infer --checkpoint runs/smoke/best.pt --data data/smoke --output runs/smoke/inference.json
pytest -q
```

The generator renders procedural RGB kinematic scenes. It is an integration fixture,
**not contact physics or LabUtopia training data**. Real recordings use the validated
NPZ import/export API documented in [DATASET.md](docs/DATASET.md). Model supervision
is physically separated from its sensor-only input dataclass.

For a longer bounded development experiment, use `configs/pilot.yaml`. Full architecture
defaults are in `configs/default.yaml` and `ModelConfig`; choose an explicit `cuda:N`
only after checking available workstation resources. Dataset and run folders are never
overwritten implicitly. No training daemon is started by installation.

## What is configurable

Image/patch sizes; vision and context depth/width; camera count; slot capacity;
joint mode count and covariance rank; time bins; history budgets; action/robot/task
dimensions; loss weights; optimizer; seed; task prior and split proportions. Unknown
config keys fail early. A new sensor, task prior or controller can be introduced behind
the typed boundaries without rewriting the shared inference core.

Checkpoints contain resolved configuration, optimizer, RNG, EMA visual teacher, source
and dataset fingerprints. `--resume` preserves the exact next optimizer step; changing
source/data/config requires a new experiment. `--init-from` starts a new optimization
run from an architecture-compatible native checkpoint.

## Research lineage and pretrained weights

This project is **PFN-inspired amortized inference**, with hidden-state readout in
addition to posterior prediction. It is not the TabPFN estimator and does not claim
exact Bayesian posteriors. See [PFN](https://arxiv.org/abs/2112.10510),
[TabPFN](https://www.nature.com/articles/s41586-024-08328-6),
[TabICL](https://arxiv.org/abs/2502.05564),
[Slot Attention](https://arxiv.org/abs/2006.15055), and
[SAVi](https://arxiv.org/abs/2111.12594).

No upstream checkpoint is bundled or falsely presented as loaded. The selected
256-dimensional core is incompatible with the inspected 1024-dimensional TabPFN-3.5
ICL weights. [Pretraining notes](docs/PRETRAINING.md) describe the mismatch and the
separate experiment needed for genuine upstream initialization.

## Repository map

```text
configs/                  full, pilot and integration configurations
src/rch_pfn/config.py      strict experiment schema
src/rch_pfn/schema.py      evidence, targets and joint belief contracts
src/rch_pfn/model.py       visual slots, shared core, conditioning and graph distributions
src/rch_pfn/losses.py      joint NLL, missing marginals, vision and prediction scores
src/rch_pfn/data/          priors, procedural rendering, safe storage, import, split audit
src/rch_pfn/runner.py      training, validation, checkpoint/resume and test evaluation
src/rch_pfn/inference.py   fixed-weight streaming and inspectable belief export
src/rch_pfn/geometry.py    rigid transformations
src/rch_pfn/control.py     one-step control contract and explicit dry-run backend
tests/                    numerical, causal, data and control regressions
docs/                     architecture, data, protocol, limitations and evidence
```

Publication and licensing metadata must be finalized by the authors before public
release. No GitHub repository, paper submission or results claim is created by this
package. Original code rights are reserved pending that decision; see `LICENSE`.
