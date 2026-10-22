"""Storage arithmetic and run-length encoding.

Byte counts here are exact for the stated uncompressed raster. A compressed
count is ``uncompressed / ratio`` and is not a measured file size.
"""

from __future__ import annotations

from collections.abc import Sequence


def storage_bits(
    width: int,
    height: int,
    bit_depth: int,
    channels: int,
    frames: int,
) -> int:
    if min(width, height, bit_depth, channels, frames) < 1:
        raise ValueError("width, height, bit depth, channels, and frames must be positive.")
    return int(width) * int(height) * int(bit_depth) * int(channels) * int(frames)


def storage_report(
    width: int,
    height: int,
    bit_depth: int,
    channels: int,
    frames: int,
    compression_ratio: float,
    transfer_bits_per_second: float,
) -> dict[str, float]:
    """Uncompressed size, ratio-based compressed size, and transfer time.

    ``compression_ratio`` means uncompressed bits / compressed bits.
    Binary units (KiB) and decimal units (MB) are both returned and named.
    """
    if compression_ratio <= 0:
        raise ValueError("compression ratio must be positive.")
    if transfer_bits_per_second <= 0:
        raise ValueError("transfer rate must be positive.")
    bits = storage_bits(width, height, bit_depth, channels, frames)
    compressed_bits = bits / float(compression_ratio)
    return {
        "uncompressed_bits": float(bits),
        "uncompressed_bytes": bits / 8.0,
        "uncompressed_kib": bits / 8.0 / 1024.0,
        "uncompressed_mb": bits / 8.0 / 1_000_000.0,
        "compressed_bits": compressed_bits,
        "compressed_bytes": compressed_bits / 8.0,
        "compressed_kib": compressed_bits / 8.0 / 1024.0,
        "compressed_mb": compressed_bits / 8.0 / 1_000_000.0,
        "transfer_seconds": compressed_bits / float(transfer_bits_per_second),
    }


def rle_encode(sequence: Sequence[float]) -> list[tuple[float, int]]:
    runs: list[tuple[float, int]] = []
    for value in sequence:
        value = float(value)
        if not runs or runs[-1][0] != value:
            runs.append((value, 1))
        else:
            symbol, count = runs[-1]
            runs[-1] = (symbol, count + 1)
    return runs


def rle_decode(runs: Sequence[tuple[float, int]]) -> list[float]:
    decoded: list[float] = []
    for value, count in runs:
        decoded.extend([float(value)] * int(count))
    return decoded


def rle_ratio(sequence: Sequence[float]) -> float:
    """Symbols in the sequence divided by symbols in (value, count) pairs.

    Each run is counted as two stored symbols. This is an accounting model,
    not a file format.
    """
    runs = rle_encode(sequence)
    if not runs:
        return 1.0
    return len(sequence) / (2.0 * len(runs))
