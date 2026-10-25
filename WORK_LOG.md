# RCH-PFN implementation work log

2026-10-25: root takes ownership; no Astra Orchestrator. Latest v2 visual-context
specification recovered. Scope: runnable modular research package, configurable
model/data/training/evaluation/inference, causal data contract, control handoff,
tests and publication-oriented documentation. Existing demonstrations remain evidence
for their own scripted pipeline, not learned-model results. Local isolated environment
created; NVIDIA RTX 5090 Laptop GPU detected. Root writes model and integrates all work.

Research design reference: visual slots -> relational action/result context -> shared
two-pass inference -> hypothesis-conditioned history rereading -> joint scene density
and action consequences -> versioned control request. Exact pretrained reuse is being
checked against upstream code. No published efficacy claim is made.

User correction: ALL training, tests and future runtime belong on the existing HP
workstation, never the local RTX 5090. No local training was started. Local CUDA
dependency download had already completed when cancellation reached it; the second
dependency installation was cancelled. Local files are a source staging/delivery mirror
only. Canonical project: /home/yxy/data/projects/rch-pfn on hp-workstation.
2026-10-25 live inventory: two RTX 8000s occupied by X-VLA and ComfyUI respectively;
do not terminate either. Initial bounded engineering checks use workstation CPU.

Implementation completed: configurable visual slots, relation contexts, shared two-pass
core/graph decoder, mode-specific pose/time/reconstruction history conditions, joint
low-rank state densities, action forecasts, masked supervision, deterministic dataset
storage/splits, recorder interchange, exact resume, streaming inference and dry-run
control contracts. Native code plus docs/configs/CI/packaging are source-controlled.

Final verification: 39 pytest tests, Ruff, full architecture forward (33,506,167 params,
1122 edges) passed on workstation CPU. Independent reviewer closed all discovered
P1/P2 issues after targeted regression tests. Final pilot-v2: 90 synthetic records,
100 optimizer steps, validation/test/inference completed; logs, datasets and weights
remain on workstation. Test coordinate RMSE about 0.037 m is an engineering fixture
diagnostic, not evidence of precision insertion. No calibrated physical-control claim.

Next research work: validated LabUtopia RGB/execution recorder, domain-specific priors,
official-pretrained transfer as a separate compatible architecture, calibration and
matched baselines/physical rollouts. No large production training job remains running.
Original LabUtopia/X-VLA/ComfyUI work was not changed.

Local virtual-environment cleanup was attempted after migration but automatically
rejected by execution policy (no specific reason returned). The inactive local .venv
remains excluded from Git and archives; no local model run occurred.
