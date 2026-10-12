# Complete ReWM training and ablations

The complete encoder is implemented in `src/moda_module.py`.
The public entry point is `scripts/train.py`; `configs/full/` contains the five selected task recipes and their schedule metadata.
These are the recipes for the reported selected runs, including the original seed and schedule. Each trajectory corresponds to one run.
It preserves the original 8-GPU recipes. Changing device count, local batch size, or gradient accumulation is not automatically treated as an equivalent run.
In particular, the SIGReg loss and projector BatchNorm operate on local batches.

```bash
# Resolve a run without reading data or starting CUDA.
python scripts/train.py --task reacher --arm full --stop-after 8 --config-only

# Training; the original two-epoch scheduler is retained when stopping at update 8.
python scripts/train.py --task reacher --arm full \
  --data-root /path/to/datasets --dataset-name dmc/reacher_random \
  --output runs/reacher_full_seed0 --stop-after 8 --save-steps 0,8

# Paired whole-module ablation, initialized from the preceding Full step-0 object.
python scripts/train.py --task reacher --arm full_no_moda_qknorm \
  --data-root /path/to/datasets --dataset-name dmc/reacher_random \
  --output runs/reacher_no_moda_seed0 --stop-after 8 --save-steps 0,8 \
  --anchor runs/reacher_full_seed0/lewm_full_ablation_step_0_object.ckpt
```

Use plain `python`, not `torchrun`: Lightning launches the eight DDP workers.
Use a new output directory for each run. The release CLI deliberately does not resume a partial run implicitly.
`--seed` selects a new encoder seed; keep the same seed and Full step-0 anchor for the paired arms.
If no anchor is supplied, the world model is initialized from the specified seed.
Historical initialization objects are supplied separately. Use the original step-0 anchor when replaying a historical initialization.
Only load model-object checkpoints from a trusted source: they use Python pickle.

| Arm | Change relative to Full |
|---|---|
| `full` | Complete ReWM |
| `full_no_moda_qknorm` | Remove the complete MoDA module, including its QKNorm |
| `full_abspos` | Replace RoPE with learned absolute positions |
| `full_gelu_mlp` | Replace SwiGLU with a parameter-matched GELU FFN |
| `full_layernorm` | Replace block and terminal RMSNorm with bias-free LayerNorm |
| `serial_multilevel_mean` | Whole MoDA module removed; mean normalized summary tokens from blocks 3/6/9/12 |
| `serial_multilevel_concat` | Whole MoDA module removed; concatenate those tokens, then project 768 to 192 |

The concatenation projection starts as four identity blocks scaled by 1/4 and adds 147,456 parameters.
Full, the mean arm, and the parameter-matched GELU arm have 5,427,841 registered encoder parameters.
The absolute-position arm adds 49,344 positional parameters.
Some disabled-module parameters remain registered, as in the experiment implementation.

| Task | Seed | Global batch | LR | WD | Original schedule epochs | Updates / epoch | Warmup updates | Default stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reacher | 0 | 512 | 2e-4 | 1e-3 | 2 | 3199 | 63 | 6398 |
| TwoRoom | 0 | 512 | 7e-4 | 1e-3 | 1 | 1284 | 12 | 512 |
| Cube | 60106 | 896 | 5e-4 | 2e-3 | 100 | 1828 | 8 | 512 |
| Puzzle 3×3 | 0 | 512 | 2e-4 | 1e-3 | 10 | 1726 | 172 | 17260 |
| PushT | 3072 | 512 | 2.5e-4 | 1e-3 | 8 | 3483 | 278 | 27864 |

`--stop-after` stops through a callback; it does not replace `max_epochs` or shorten the cosine schedule.
The defaults above are Full recipe budgets, not a claim that every ablation was evaluated for that entire duration.
For the five-task early mean/concat comparison, use `--stop-after 8 --save-steps 0,8` with the corresponding Full step-0 anchor and seed.
Report each arm's measured checkpoint grid separately from its selected-run peak statistics.
Cube uses the recorded warmup/hold/decay schedule: warmup through update 8, hold through 128, cosine decay to 0.5× LR at 512.
The other tasks use linear warmup and cosine decay over the original schedule horizon.
In stable-pretraining 0.1.6, manual optimization advances the scheduler per optimizer update, including the recipes whose serialized scheduler interval is `epoch`.
The release preserves these settings and checks the realized LR on every update.
The runtime also verifies the expected data cardinality, global batch, and schedule horizon, and fails rather than silently running a different protocol.

HDF5 dataset names are relative to `--data-root`, without `.h5`:
`reacher`, `tworoom`, `ogbench/cube_single_expert`, `ogbench/puzzle3x3_train`, and `pusht_expert_train`.
`--dataset-name` changes a local filename, not data content.
The HDF5 schema must match stable-worldmodel 0.0.6: episode `ep_len` and `ep_offset` metadata plus the configured pixel/action/observation columns.
The fixed data cardinalities matter for exact schedule parity.

Outputs include step-0 and requested update model objects, complete-epoch objects, a final Lightning checkpoint, resolved config, tensor initialization hashes, LR trace, and `TRAIN_COMPLETE.json`.
The historical module names (`moda_module`, `jepa`, `module`, `full_ablation`, `multilevel_mean`, `multilevel_concat`) are kept so model-object checkpoints can be imported with `src` on `PYTHONPATH`.

Validation in this packaging pass: Python syntax, CLI help, five resolved recipe consistency checks, all seven arm constructions and parameter counts, and Cube scheduler values at updates 0/8/128/512.
For all seven arms, the packaged and original models also produced bit-identical initialized state dictionaries, outputs, and gradients on a synthetic 56×56 GPU forward/backward check.
This is packaging validation; no full training rerun was performed by this pass.

Official and standard-block comparisons use `scripts/train_comparison.py`. The official entry resolves the shipped Hydra config directory and retains the archived official training source unchanged. Official runs default to seed 3072, global batch 128, LR 5e-5, and 100 epochs; the controlled pair uses seeds 78101/78102/78103, global batch 896, LR 2e-4, 16 warmup updates, and the original 100-epoch horizon. Pass `--seed` for each controlled repeat. These families have distinct training and evaluation protocols; see `docs/experiment_map.md`.

The reported standard-block paired comparison covers Reacher and TwoRoom. The launcher also accepts other task datasets, but running those combinations creates additional experiments; it does not establish a reported paired result.
