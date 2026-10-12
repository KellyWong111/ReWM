#!/usr/bin/env python3
"""Full and official-model diagnostics in eval mode, using the fixed data bank.

Encoder weights and BatchNorm buffers stay frozen. Extra image indices
preserve the historical encoder batching and feature alignment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ['TRITON_F32_DEFAULT'] = 'ieee'
os.environ.setdefault('FLA_LIGHT_IMPORT', '1')
import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch
from raw_common import (sha_file, dump, tensor_hash, bn_buffers, targets,
                        regression_metrics, fit_ridge, decode, native_prediction_metrics)

def register_readouts(encoder, architecture):
    features = {}; handles = []
    if architecture == 'full':
        if not hasattr(encoder, 'transformer') or not hasattr(encoder.transformer, 'layers'):
            raise TypeError('Expected Full MoDAVisualEncoder')
        layers = encoder.transformer.layers; index = -1
        if not encoder.transformer.use_moda or not all(b.attn.qk_norm for b in layers):
            raise ValueError('Full checkpoint must retain integrated MoDA+QKNorm')
        normalize = lambda x: encoder.transformer.output_proj(encoder.transformer.norm(x))
    elif architecture in ('official', 'official_vit'):
        hf = encoder.vit if hasattr(encoder, 'vit') else encoder
        if not hasattr(hf, 'encoder') or not hasattr(hf.encoder, 'layer'):
            raise TypeError('Expected official HF ViT encoder')
        if hasattr(encoder, 'use_moda') and encoder.use_moda:
            raise ValueError('Official baseline unexpectedly enables MoDA')
        layers = hf.encoder.layer; index = 0; normalize = hf.layernorm
    else:
        raise ValueError(architecture)
    if len(layers) != 12:
        raise ValueError(f'Expected 12 encoder blocks, got {len(layers)}')
    for layer, block in enumerate(layers, 1):
        if layer in (3, 6, 9, 12):
            def capture(_m, _args, output, layer=layer):
                h = output[0] if isinstance(output, (tuple, list)) else output
                features[f'raw_l{layer}'] = h[:, index].detach()
            handles.append(block.register_forward_hook(capture))
    return features, handles, normalize, index


def action_windows(h5, bank, mean, std):
    result = []
    tm, ts = torch.from_numpy(mean), torch.from_numpy(std)
    for global_start in bank[:, -1]:
        raw = torch.from_numpy(np.asarray(h5['action'][int(global_start):int(global_start + 20)]))
        # Preserve original normalization dtype and ordering, especially Reacher FP64.
        if raw.dtype != tm.dtype or raw.shape[0] != 20:
            raise ValueError('Action dtype/window length differs from frozen bank')
        normalized = ((raw - tm) / ts).float().reshape(4, -1)
        result.append(torch.nan_to_num(normalized, nan=0.0))
    return torch.stack(result)


def predict_windows(model, z, actions, device):
    predictions = []
    for start in range(0, len(z), 64):
        emb = torch.from_numpy(z[start:start + 64]).to(device)
        act = model.action_encoder(actions[start:start + 64].to(device))
        pred = model.predict(emb[:, :3], act[:, :3]).float()
        if pred.shape != emb[:, 1:].shape:
            raise ValueError('Prediction alignment is not original next-step sequence alignment')
        predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions)


def persistent_buffers(model):
    parameter_names = set(dict(model.named_parameters()))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in parameter_names}


def run(a):
    case = json.loads(Path(a.case).read_text())
    if case['task'] not in ('reacher', 'tworoom', 'pusht'):
        raise ValueError('Supported tasks: Reacher, TwoRoom, PushT')
    out, bank_dir = Path(a.output), Path(a.bank); out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((bank_dir / 'metadata.json').read_text())
    bank_sha = sha_file(bank_dir / 'bank.npz')
    if bank_sha != metadata['bank_sha256'] or case['task'] != metadata['spec']['task']:
        raise ValueError('Frozen bank identity/task mismatch')
    checkpoint_sha = sha_file(case['checkpoint'])
    if checkpoint_sha != case['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    code_hash, common_hash = sha_file(__file__), sha_file(Path(__file__).with_name('raw_common.py'))
    model_sources = {path.name: sha_file(path) for path in sorted(Path(case['model_code']).glob('*.py'))}
    kernel_root = Path(case['kernel_root']) if case.get('kernel_root') else None
    kernel_sources = {str(path.relative_to(kernel_root)): sha_file(path)
                      for path in sorted(kernel_root.rglob('*.py'))} if kernel_root else {}
    versions = {}
    for name in ('torch', 'numpy', 'triton', 'h5py', 'transformers', 'stable-worldmodel', 'stable-pretraining'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    dataset_stat = Path(metadata['spec']['h5']).stat()
    identity = dict(case_id=case['id'], task=case['task'], architecture=case['architecture'],
                    checkpoint_sha256=checkpoint_sha, bank_sha256=bank_sha,
                    bank_metadata_sha256=sha_file(bank_dir / 'metadata.json'),
                    source_code_sha256=code_hash, common_code_sha256=common_hash,
                    model_code_sha256=model_sources, kernel_source_sha256=kernel_sources,
                    dependency_versions=versions, encoder_batch_size=a.batch_size,
                    dataset_file_identity=dict(path=str(Path(metadata['spec']['h5']).resolve()),
                                               bytes=dataset_stat.st_size, mtime_ns=dataset_stat.st_mtime_ns))
    if (out / 'COMPLETE.json').exists() and a.smoke:
        raise ValueError('Smoke output cannot overwrite a completed formal diagnostic; choose a fresh directory')
    if (out / 'COMPLETE.json').exists() and not a.smoke:
        old = json.loads((out / 'COMPLETE.json').read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing completion has a different identity or source')
        for name, digest in old['artifacts_sha256'].items():
            if sha_file(out / name) != digest:
                raise ValueError('Completed artifact changed: ' + name)
        print('SKIP_VERIFIED_COMPLETE', case['id'], flush=True); return
    if any((out / name).exists() for name in ('METRICS.json', 'SMOKE_METRICS.json', 'arrays.npz')):
        raise ValueError('Existing diagnostic artifacts cannot be overwritten; choose a fresh output directory')
    sys.path.insert(0, case['model_code'])
    if case.get('kernel_root'):
        os.environ['MODA_TRITON_ROOT'] = case['kernel_root']; sys.path.insert(1, case['kernel_root'])
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260925); started = time.time()
    loaded = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
    model = loaded.model if hasattr(loaded, 'model') else loaded
    unexpected = {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    if unexpected:
        raise ValueError(f'Archived learned parameters are not FP32: {unexpected}')
    model = model.eval().to(a.device).requires_grad_(False)
    if torch.is_autocast_enabled() or any(m.training for m in model.modules()):
        raise AssertionError('Requires frozen original eval with autocast disabled')
    learned_hash = tensor_hash(model.named_parameters()); original_bn = bn_buffers(model)
    original_bn_hash = tensor_hash(original_bn.items()); original_buffers_hash = tensor_hash(persistent_buffers(model).items())
    bns = {name for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    if bns != {'projector.net.1', 'pred_proj.net.1'}:
        raise ValueError(f'Unexpected BatchNorm modules {bns}')
    bank = np.load(bank_dir / 'bank.npz')
    probe_bank, pred_bank = bank['probe_bank'], bank['prediction_bank']
    # Retain these old extra image rows ONLY to preserve encoder batch layout.
    # No statistics are estimated, no buffers are updated, and no extra scores
    # are produced from them. All requested rows are the original common bank.
    alignment_rows = bank['calibration_bank']
    if a.smoke:
        probe_bank = np.concatenate([probe_bank[probe_bank[:, 0] == s][:16 if s == 0 else 8] for s in (0, 1, 2)])
        pred_bank, alignment_rows = pred_bank[:8], alignment_rows[:16]
    train, validation, test = [probe_bank[:, 0] == s for s in (0, 1, 2)]
    for x, y in ((train, validation), (train, test), (validation, test)):
        if not set(probe_bank[x, 2]).isdisjoint(probe_bank[y, 2]):
            raise ValueError('Readout episode leakage')
    probe_frames = probe_bank[:, 4] + 10
    pred_frames = pred_bank[:, 3, None] + np.array([0, 5, 10, 15])
    alignment_frames = alignment_rows[:, 3, None] + np.array([0, 5, 10, 15])
    unique = np.unique(np.concatenate([probe_frames, pred_frames.reshape(-1), alignment_frames.reshape(-1)]))
    probe_map, pred_map = np.searchsorted(unique, probe_frames), np.searchsorted(unique, pred_frames)
    captured, handles, normalize_final, internal_summary_index = register_readouts(model.encoder, case['architecture'])
    feature_rows = {key: [] for key in ('raw_l3', 'raw_l6', 'raw_l9', 'raw_l12', 'postnorm_l12')}
    pixel_hash = hashlib.sha256(); max_final_error = 0.0
    with h5py.File(metadata['spec']['h5'], 'r') as f, torch.inference_mode():
        if hashlib.sha256(f['ep_len'][:].tobytes()).hexdigest() != metadata['episode_lengths_sha256'] or hashlib.sha256(f['ep_offset'][:].tobytes()).hexdigest() != metadata['episode_offsets_sha256']:
            raise ValueError('Dataset episode metadata changed')
        probe_targets = targets(f, probe_frames, case['task'])
        pred_actions = action_windows(f, pred_bank, bank['action_mean'], bank['action_std'])
        mean = torch.tensor([.485, .456, .406], device=a.device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=a.device).view(1, 3, 1, 1)
        for start in range(0, len(unique), a.batch_size):
            pix = np.asarray(f['pixels'][unique[start:start + a.batch_size]])
            if pix.dtype != np.uint8 or pix.shape[1:] != (224, 224, 3):
                raise ValueError('Expected original 224x224 RGB uint8 observations')
            pixel_hash.update(pix.tobytes())
            images = torch.from_numpy(pix).to(a.device).float().permute(0, 3, 1, 2) / 255
            captured.clear(); encoded = model.encoder((images - mean) / std, interpolate_pos_encoding=True)
            final = encoded.last_hidden_state[:, 0]; expected = normalize_final(captured['raw_l12'])
            torch.testing.assert_close(final, expected, atol=5e-5, rtol=1e-5)
            max_final_error = max(max_final_error, float((final - expected).abs().max()))
            for key, value in {**captured, 'postnorm_l12': final}.items():
                if value.shape != (len(pix), 192) or not torch.isfinite(value).all():
                    raise ValueError('Nonfinite/unexpected encoder output ' + key)
                feature_rows[key].append(value.float().cpu().numpy())
            if start % (a.batch_size * 50) == 0:
                print('ENCODE', case['id'], start, len(unique), 'seconds', round(time.time() - started, 1), flush=True)
        for handle in handles:
            handle.remove()
        features = {key: np.concatenate(rows) for key, rows in feature_rows.items()}
        hidden = torch.from_numpy(features['postnorm_l12'])
        raw_z = torch.cat([model.projector(h.to(a.device)).float().cpu() for h in hidden.split(128)]).numpy()
        raw_sequence = raw_z[pred_map]
        raw_prediction = predict_windows(model, raw_sequence, pred_actions, a.device)
        raw_metrics = native_prediction_metrics(raw_sequence, raw_prediction)
    if tensor_hash(model.named_parameters()) != learned_hash:
        raise AssertionError('Learned parameters changed')
    if tensor_hash(bn_buffers(model).items()) != original_bn_hash:
        raise AssertionError('Original BatchNorm buffers changed')
    if tensor_hash(persistent_buffers(model).items()) != original_buffers_hash:
        raise AssertionError('Checkpoint persistent buffers changed')
    readouts = {key: value[probe_map] for key, value in features.items()}
    readouts['z_original_eval'] = raw_z[probe_map]
    fit_results = {}; arrays = dict(probe_bank=probe_bank, prediction_bank=pred_bank,
                                    probe_frames=probe_frames, prediction_frames=pred_frames,
                                    raw_prediction=raw_prediction, raw_sequence=raw_sequence)
    for name, x in readouts.items():
        fit_results[name] = {}
        for target, y in probe_targets.items():
            fit = fit_ridge(x, y, train, validation)
            fit_results[name][target] = dict(**regression_metrics(decode(x[test], fit), y[test], fit['ym'], fit['ys']),
                                             ridge_alpha=fit['alpha'], validation_nmse=fit['validation_nmse'])
        arrays[name] = x
    arrays.update({'probe_target_' + key: value for key, value in probe_targets.items()})
    np.savez_compressed(out / 'arrays.npz', **arrays)
    source_files = model_sources
    report = dict(case=case, case_id=case['id'], task=case['task'], architecture=case['architecture'], smoke=a.smoke,
                  checkpoint_sha256=checkpoint_sha, bank_sha256=bank_sha, bank_metadata_sha256=sha_file(bank_dir / 'metadata.json'),
                  source_code_sha256=code_hash, common_code_sha256=common_hash, model_code_sha256=source_files,
                  encoder_batch_size=a.batch_size, dataset_file_identity=identity['dataset_file_identity'],
                  kernel_source_sha256=kernel_sources, dependency_versions=versions,
                  dataset_identity_scope='File location/stat guard; selected pixels/targets/actions hashed below, not whole-dataset identity',
                  data_scope=metadata['scope'], action_normalization_scope=metadata['action_normalization_scope'],
                  action_statistics_dtype=str(bank['action_mean'].dtype),
                  preprocessing='Original RGB224 ImageNet normalization in FP32; action normalization in original dtype then float32',
                  precision='Native FP32 original eval, dropout off, PyTorch TF32/autocast off, Triton IEEE',
                  internal_summary_index=internal_summary_index, returned_summary_index=0, final_normalization_check_max_abs=max_final_error,
                  encoder_class=type(model.encoder).__module__ + '.' + type(model.encoder).__name__, encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                  original_bn_sha256=original_bn_hash, probe_prediction_maps=dict(original_eval=original_bn_hash),
                  persistent_buffers_sha256=original_buffers_hash, buffers_unchanged=True,
                  learned_parameters_sha256=learned_hash, weights_unchanged=True,
                  selected_pixels_sha256=pixel_hash.hexdigest(), frame_indices_sha256=hashlib.sha256(unique.tobytes()).hexdigest(),
                  probe_target_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in probe_targets.items()},
                  prediction_action_sha256=hashlib.sha256(pred_actions.numpy().tobytes()).hexdigest(),
                  sample_counts=dict(probe_fit=int(train.sum()), probe_validation=int(validation.sum()), probe_test=int(test.sum()),
                                     prediction_windows=len(pred_bank), unique_images=len(unique)),
                  bank_policy='Original frozen window/episode selections and encoder image batch layout reused; extra images affect neither statistics nor reported target selection.',
                  probes=fit_results, native_prediction=dict(original_eval=raw_metrics),
                  diagnostic_scope='Six requested original-eval readouts and own-latent prediction only. No BN statistics fitting/update, encoder training, controller fitting, or additional prediction experiment.',
                  seconds=time.time() - started)
    report['runtime_kernel_sources'] = {name: dict(path=str(module.__file__), sha256=sha_file(module.__file__))
                                        for name, module in list(sys.modules.items())
                                        if name.startswith('fla.ops.moda') and getattr(module, '__file__', None) and Path(module.__file__).is_file()}
    metric_name = 'SMOKE_METRICS.json' if a.smoke else 'METRICS.json'
    dump(out / metric_name, report)
    if not a.smoke:
        dump(out / 'COMPLETE.json', dict(**identity,
                                       metrics_path=str(out / metric_name), native_prediction=report['native_prediction'], weights_unchanged=True,
                                       buffers_unchanged=True, seconds=report['seconds'],
                                       artifacts_sha256={name: sha_file(out / name) for name in ('METRICS.json', 'arrays.npz')}))
    print('COMPLETE', case['id'], 'smoke', a.smoke, 'seconds', round(report['seconds'], 1), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--case', required=True); p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--smoke', action='store_true'); run(p.parse_args())


if __name__ == '__main__':
    main()
