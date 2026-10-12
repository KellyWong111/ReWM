"""Shared numerics for frozen Full/official-ViT diagnostics; no model imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def bn_buffers(model):
    return {name + '.' + key: value.detach().cpu().clone()
            for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def targets(h5, idx, task):
    # h5py requires sorted unique indices; caller restores its original order.
    idx, inverse = np.unique(np.asarray(idx), return_inverse=True)
    if task == 'reacher':
        fp, tp, q = h5['finger_pos'][idx], h5['target_pos'][idx], h5['qpos'][idx]
        result = dict(finger_position=fp, target_position=tp,
                      joint_angles_sincos=np.concatenate([np.sin(q), np.cos(q)], axis=-1), interaction_geometry=tp - fp)
    elif task == 'tworoom':
        ag, tg = h5['pos_agent'][idx], h5['pos_target'][idx]
        result = dict(agent_position=ag, target_position=tg, interaction_geometry=tg - ag)
    elif task == 'pusht':
        s = h5['state'][idx]; yaw = s[:, 4]
        result = dict(pusher_position=s[:, :2], object_position=s[:, 2:4],
                      object_orientation_sincos=np.stack([np.sin(yaw), np.cos(yaw)], axis=-1),
                      interaction_geometry=s[:, 2:4] - s[:, :2])
    else:
        raise ValueError(task)
    return {k: np.asarray(v[inverse], np.float32) for k, v in result.items()}


def regression_metrics(pred, truth, ym, ys):
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    sse = np.square(pred - truth).sum(0); sst = np.square(truth - truth.mean(0)).sum(0); valid = sst > 1e-12
    return dict(mse=float(np.square(pred - truth).mean()), normalized_mse=float(np.square((pred - truth) / ys).mean()),
                r2_uniform_average=float(np.mean(1 - sse[valid] / sst[valid])) if valid.any() else None,
                r2_variance_weighted=float(1 - sse.sum() / sst.sum()) if sst.sum() > 1e-12 else None,
                constant_test_dimensions=int((~valid).sum()), target_dimensions=truth.shape[1],
                train_mean_baseline_nmse=float(np.square((truth - ym) / ys).mean()))


def fit_ridge(x, y, train, validation):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xm = x[train].mean(0); xs = np.maximum(x[train].std(0), 1e-8)
    ym = y[train].mean(0); ys = np.maximum(y[train].std(0), 1e-8)
    xt, yt = torch.from_numpy((x[train] - xm) / xs), torch.from_numpy((y[train] - ym) / ys)
    xv, yv = (x[validation] - xm) / xs, (y[validation] - ym) / ys
    gram, rhs = xt.T @ xt / len(xt), xt.T @ yt / len(xt)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype); best = None
    for alpha in (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.):
        w = torch.linalg.solve(gram + alpha * eye, rhs).numpy()
        loss = float(np.square(xv @ w - yv).mean())
        if best is None or loss < best['validation_nmse']:
            best = dict(w=w, alpha=alpha, validation_nmse=loss)
    return dict(**best, xm=xm, xs=xs, ym=ym, ys=ys)


def decode(x, fit):
    return ((np.asarray(x, np.float64) - fit['xm']) / fit['xs'] @ fit['w']) * fit['ys'] + fit['ym']


def native_prediction_metrics(z, pred):
    target = np.asarray(z[:, 1:], np.float64); pred = np.asarray(pred, np.float64)
    assert pred.shape == target.shape and np.isfinite(pred).all() and np.isfinite(target).all()
    flat = target.reshape(-1, target.shape[-1]); center = flat - flat.mean(0)
    sse = float(np.square(pred - target).sum()); sst = float(np.square(center).sum())
    eig = np.maximum(np.linalg.eigvalsh(center.T @ center / len(flat)), 0)
    probabilities = eig / max(float(eig.sum()), 1e-30); positive = probabilities > 0
    return dict(raw_mse=float(np.square(pred - target).mean()), r2=1 - sse / sst if sst > 1e-15 else None,
                target_total_variance=sst / len(flat), effective_rank=float(np.exp(-(probabilities[positive] * np.log(probabilities[positive])).sum())),
                n_sequences=len(z), predicted_positions=3, target='each encoder own latent in the named normalization mode; not a common-target accuracy measure')
