#!/usr/bin/env python3
"""Convert OGBench Puzzle NPZ to the historical training HDF5 format.

Repeat 64x64 pixels 3x, then edge-pad 16 pixels per side to reach 224x224.
No rendering is performed.
"""
import argparse
from pathlib import Path
import zipfile

import h5py
import numpy as np


def _header(stream):
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    if dtype.hasobject:
        raise ValueError("Object arrays are not supported; pickle loading is disabled.")
    return shape, fortran, dtype


def convert(input_path, output_path, chunk_frames=2000):
    if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or chunk_frames <= 0:
        raise ValueError("chunk_frames must be a positive integer")
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    with zipfile.ZipFile(input_path) as archive:
        # Read sequentially to avoid decompressing earlier frames again.
        with archive.open("observations.npy") as pixels:
            shape, fortran, dtype = _header(pixels)
            if len(shape) != 4 or shape[0] <= 0 or shape[1:] != (64, 64, 3):
                raise ValueError("observations must have shape (N, 64, 64, 3), with N > 0")
            if fortran or dtype != np.dtype(np.uint8):
                raise ValueError("observations must be C-order uint8 pixels")
            n_frames = shape[0]
            row_bytes = 64 * 64 * 3
            if archive.getinfo("observations.npy").file_size != pixels.tell() + n_frames * row_bytes:
                raise ValueError("observations payload length does not match its header")

            arrays = {}
            for name in ("terminals", "actions", "qpos", "qvel"):
                with archive.open(name + ".npy") as stream:
                    small_shape, _, small_dtype = _header(stream)
                    required_ndim = 1 if name == "terminals" else 2
                    if (len(small_shape) != required_ndim or small_shape[0] != n_frames
                            or (required_ndim == 2 and small_shape[1] <= 0)):
                        raise ValueError(f"{name} has an invalid shape or frame count: {small_shape}")
                    if small_dtype.kind not in "biuf":
                        raise ValueError(f"{name} must have a real numeric or boolean dtype")
                with archive.open(name + ".npy") as stream:
                    arrays[name] = np.load(stream, allow_pickle=False)

            terms = arrays["terminals"].astype(bool)
            actions = arrays["actions"].astype(np.float32)
            # Match the historical float32 rounding before conversion to float64.
            qpos = arrays["qpos"].astype(np.float32)
            qvel = arrays["qvel"].astype(np.float32)
            observation = np.concatenate([qpos, qvel], axis=1).astype(np.float64)

            ends = np.where(terms)[0]
            offsets, lengths = [], []
            start = 0
            for end in ends:
                offsets.append(start)
                lengths.append(end - start + 1)
                start = end + 1
            if start < n_frames:
                offsets.append(start)
                lengths.append(n_frames - start)
            ep_offset = np.array(offsets, dtype=np.int64)
            ep_len = np.array(lengths, dtype=np.int32)
            ep_idx = np.empty(n_frames, dtype=np.int32)
            for index, (offset, length) in enumerate(zip(offsets, lengths)):
                ep_idx[offset:offset + length] = index

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against races after exists().
            with h5py.File(output_path, "x") as target:
                output_pixels = target.create_dataset(
                    "pixels", shape=(n_frames, 224, 224, 3), dtype=np.uint8,
                    chunks=(1, 224, 224, 3),
                )
                target.create_dataset("action", data=actions, dtype=np.float32)
                target.create_dataset("observation", data=observation, dtype=np.float64)
                target.create_dataset("ep_idx", data=ep_idx, dtype=np.int32)
                target.create_dataset("ep_len", data=ep_len, dtype=np.int32)
                target.create_dataset("ep_offset", data=ep_offset, dtype=np.int64)
                written = 0
                while written < n_frames:
                    count = min(chunk_frames, n_frames - written)
                    raw = pixels.read(count * row_bytes)
                    if len(raw) != count * row_bytes:
                        raise ValueError("Unexpected end of observations payload")
                    chunk = np.frombuffer(raw, dtype=np.uint8).reshape(count, 64, 64, 3)
                    upsampled = np.repeat(np.repeat(chunk, 3, axis=1), 3, axis=2)
                    padded = np.pad(upsampled, ((0, 0), (16, 16), (16, 16), (0, 0)), mode="edge")
                    output_pixels[written:written + count] = padded
                    written += count
                    if written % (chunk_frames * 5) == 0 or written == n_frames:
                        print(f"{written}/{n_frames} frames", flush=True)
    print(f"Saved {output_path}", flush=True)


def _positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original Puzzle OGBench NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="New HDF5 path; existing files are refused.")
    parser.add_argument("--chunk-frames", type=_positive_integer, default=2000)
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.chunk_frames)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")


if __name__ == "__main__":
    main()
