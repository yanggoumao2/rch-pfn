# Initialization and post-training

The runnable branch uses independently initialized visual/ICL components followed by
joint prior-data training. Native `--init-from` supports subsequent task-domain
post-training with an identical architecture; `--resume` restores the entire run.
These two operations are deliberately different.

This does not complete the earlier proposal of reusing a specific official TabPFN
checkpoint. Source-level inspection of TabPFN-3.5 at commit
`9d9d8725168810e5f9de76bc68e0d4d6688ae26f` identifies a 1024-wide ICL core, 24 layers,
16 heads and FFN 2048. Our full core is 256-wide, 6 layers, 8 heads and FFN 1024.
Q/K normalization and attention topology also differ. Tensor interpolation, truncating
weights or changing a filename would not establish compatibility.

Candidate upstream checkpoint: `tabpfn-v3.5-20260909.safetensors`. A future genuine
transfer experiment must instantiate compatible internal ICL blocks, provide external
256→1024→256 adapters, explicitly integrate the hypothesis bias, record which layers
were inherited, and compare frozen/fine-tuned/from-scratch conditions. Such an adapter
is a distinct architecture and computational budget. This repository intentionally
fails strict native initialization on incompatible configurations.

The researcher did not download the checkpoint or inspect its safetensors header.
Key/shape observations are source-derived, not tensor-level loading evidence.
No pretrained transfer benefit has been measured. No external weights were copied.

Sources: [official architecture](https://github.com/PriorLabs/TabPFN/blob/9d9d8725168810e5f9de76bc68e0d4d6688ae26f/src/tabpfn/architectures/tabpfn_v3_5.py),
[model card](https://huggingface.co/Prior-Labs/tabpfn_3_5),
[weight license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE).
The inspected weight license has use/distribution restrictions, including derived
models; it must be reviewed before any transfer checkpoint is shared.
