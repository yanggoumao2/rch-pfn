# Original ideas, implementation choices, and unverified claims

| Original requirement | Implemented computation / boundary |
|---|---|
| Relations, not three hardcoded failure classes | Visual slots plus anchors, shared all-pairs relation encoding |
| First round already reads history | Full arrived context read in both rounds |
| Condition history on each joint explanation | Per-mode pose/time/latent condition and reconstruction residual; distinct query/bias |
| Share the PFN module | One context core and one graph decoder reused |
| Preserve uncertainty and scene dependence | Global mixture, low-rank covariance, joint samples for forecasts |
| No privileged online labels | Explicit EvidenceBatch; separate TrainingTargets and strict NPZ namespaces |
| Fixed deployment weights | Inference uses eval/no_grad; append actual observations and recompute |
| Configurable rather than 10° scripts | Width/depth/counts/prior/action schemas in config; no angle-answer constant |
| Reusable controller | Typed handoff, explicit dry-run, dependency-injected IK/collision/backend |

Engineering choices that need to stay visible:

* Segment aggregation currently uses shared endpoint feature mixing and masked segment
  pooling; it is not an independently pretrained TabICL row encoder.
* Slot correspondence is recurrent and supervised with matched masks. Full marginalization
  over arbitrary identity permutations is not implemented; explicit supplied pose
  equivalence representatives are supported. Do not report open-world identity recovery.
* The state density is a local coordinate density, not the proposal's right-tangent
  SE(3) density. Covariance labels and exports say this explicitly.
* The relation conditioner presently uses reconstruction consistency. Causal future
  prediction is trained separately, but no historical saved-prediction cache is used
  as an extra innovation claim.
* Regime-time scores are per-edge conditional scores. Repeated event IDs in source
  logs are not interpreted as independent physical faults.
* Contact and task-relevance heads have optional masked supervised losses and recorder
  fields. They require separately supplied labels before their outputs can be used.
  The procedural generator does not establish physical contact or causal relevance;
  the generic JSON export withholds these scores pending supervision/calibration provenance.
* Full identity-to-controller binding, calibrated uncertainty, camera observation
  acquisition and live LabUtopia/robot backend wiring remain deployment integrations.
  The project supplies typed boundaries and a tested dry-run backend; it does not
  pretend to have executed the learned model on the earlier robot scenario.

These distinctions prevent a runnable engineering package from being mistaken for
experimental proof. In particular, successful smoke optimization is not proof of
history reinterpretation, PFN superiority, transfer to LabUtopia, or paper novelty.
