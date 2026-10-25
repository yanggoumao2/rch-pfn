# Development conventions

Use an isolated environment on the workstation. Keep source, data and run directories
separate. Do not add private paths, credentials, simulator caches or large weights to Git.
Configuration changes that alter tensor/target semantics need a schema version change.

Before a change, read the typed contracts and its associated tests. For model changes,
check finite backward gradients, causal masks, probability normalization and the two
pass coupling. For data changes, verify hashes, group-level splits and no target leakage.
For control changes, test rejection paths and frame/identity/expiry revalidation.

Run `pytest -q` and `ruff check src tests` on the workstation. A short training round-trip
is appropriate when checkpoint, loss or model behavior changes. Do not add redundant
long training jobs for documentation-only changes.
