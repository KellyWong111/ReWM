#!/usr/bin/env python3
"""Train official-recipe LeWM or the paired standard-block models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'reacher': ('dmc', 'reacher'),
    'tworoom': ('tworoom', 'tworoom'),
    'cube': ('ogb', 'ogbench/cube_single_expert'),
    'puzzle3x3': ('ogb', 'ogbench/puzzle3x3_train'),
    'pusht': ('pusht', 'pusht_expert_train'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['official', 'controlled'], required=True)
    parser.add_argument('--arm', choices=['reference', 'lewm_moda'], default='reference')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr', default=None, help='Encoder learning rate (default 2e-4 controlled, 5e-5 official).')
    parser.add_argument('--stop-after', type=int, default=None, help='Stop a prefix without shortening the 100-epoch scheduler.')
    parser.add_argument('--save-steps', default='0,8,16,32,64,128,256,512', help='Comma-separated update checkpoints; the stop point is also saved.')
    parser.add_argument('--dry-run', action='store_true', help='Print command and settings without reading data or starting training.')
    a = parser.parse_args()
    if a.protocol == 'official' and a.arm != 'reference':
        parser.error('The official protocol uses the original LeWM encoder. Use controlled for LeWM+MoDA.')
    seed = a.seed if a.seed is not None else (3072 if a.protocol == 'official' else 78101)
    budget = a.stop_after if a.stop_after is not None else (0 if a.protocol == 'official' else 512)
    if budget < 0 or (a.protocol == 'controlled' and budget == 0):
        parser.error('controlled requires a positive prefix budget; official zero means full training.')
    try:
        save_steps = [int(s.strip()) for s in a.save_steps.split(',') if s.strip()]
    except ValueError:
        parser.error('--save-steps must contain comma-separated nonnegative integers')
    if any(s < 0 for s in save_steps):
        parser.error('save steps must be nonnegative')
    controlled = a.protocol == 'controlled'
    batch, devices = (112,8) if controlled else (128,1)
    lr = a.lr if a.lr is not None else ('2e-4' if controlled else '5e-5')
    arm = 'combo_qrm' if a.arm == 'lewm_moda' else 'combo_base'
    output, data_root = a.output.resolve(), a.data_root.resolve()
    data, name = TASKS[a.task]
    command = [sys.executable, str(ROOT/'controlled'/('train_dense.py' if controlled else 'official_entry.py')),
               f'data={data}', f'data.dataset.name={name}', f'trainer.devices={devices}',
               f'loader.batch_size={batch}', f'optimizer.lr={lr}', 'optimizer.weight_decay=1e-3',
               f'seed={seed}', f'subdir={output}', 'output_model_name=lewm', 'wandb.enabled=false']
    if a.task == 'puzzle3x3':
        command.append('~data.dataset.keys_to_merge')
    if controlled:
        command += ['+trainer.strategy=ddp_clean_timeout', '+optimizer.betas=[0.9,0.999]']
    save = sorted(set(save_steps + ([budget] if budget else []) + [0]))
    if budget:
        save = [s for s in save if s <= budget]
    env_settings = dict(STABLEWM_HOME=str(data_root), RUN_SEED=str(seed), CELL_ROOT=str(output),
                        TRAIN_BUDGET=str(budget), SAVE_STEPS=json.dumps(save), ARM=arm,
                        TRAIN_LR=lr, TRAIN_WD='1e-3', WARMUP_STEPS='16' if controlled else 'official',
                        TRAIN_EPOCH_TARGET='0', CANONICAL_INIT_CHECKPOINT='', CANONICAL_INIT_SHA256='',
                        FLA_LIGHT_IMPORT='1', MODA_TRITON_ROOT=str(ROOT/'third_party/moda_triton'),
                        PYTHONPATH=os.pathsep.join([str(ROOT/'controlled'),str(ROOT/'src'),str(ROOT/'third_party/moda_triton')]))
    plan = dict(protocol=a.protocol, arm=a.arm, task=a.task, seed=seed,
                global_batch=batch*devices, devices=devices, schedule_epochs=100,
                stop_after=budget or None, save_steps=save, command=command, environment=env_settings)
    if a.dry_run:
        print(json.dumps(plan, indent=2)); return
    expected = data_root/(name+'.h5')
    if not expected.is_file():
        parser.error(f'Dataset missing: {expected}. See docs/data.md.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output is nonempty; choose a fresh output directory to avoid overwriting a run.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'launch.json').write_text(json.dumps(plan, indent=2)+'\n')
    env = os.environ.copy(); env.update(env_settings)
    subprocess.run(command, env=env, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
