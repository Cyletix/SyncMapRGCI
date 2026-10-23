from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PACKING_FACTOR = 0.64
_KISSING_NUMBER_BY_DIM = {
    1: 2,
    2: 6,
    3: 12,
}


@dataclass(frozen=True)
class DensityScale:
    reference_local_scale: float
    effective_local_scale: float
    space_size: float
    significant_distance: float
    significant_k: int
    scale_factor: float


def kissing_number(dimensions: int) -> int:
    dim = max(1, int(dimensions))
    if dim in _KISSING_NUMBER_BY_DIM:
        return _KISSING_NUMBER_BY_DIM[dim]
    return max(2, 2 * dim)


def compute_density_scale(
    *,
    input_size: int,
    dimensions: int,
    base_wall: float,
    density_ref: float,
    significant_distance_factor: float,
    packing_factor: float = DEFAULT_PACKING_FACTOR,
    explicit_space_size: float | None = None,
) -> DensityScale:
    n_points = max(1, int(input_size))
    dim = max(1, int(dimensions))
    wall = max(float(base_wall), 1e-6)
    density = max(float(density_ref), 1e-6)
    packing = max(float(packing_factor), 1e-6)
    lambda_sig = max(float(significant_distance_factor), 1e-6)

    reference_local_scale = (wall / 2.0) * ((packing / density) ** (1.0 / dim))
    scale_factor = (n_points / density) ** (1.0 / dim)

    if explicit_space_size is not None and float(explicit_space_size) > 0.0:
        space_size = float(explicit_space_size)
        effective_local_scale = (space_size / 2.0) * ((packing / n_points) ** (1.0 / dim))
    else:
        space_size = wall * scale_factor
        effective_local_scale = reference_local_scale

    significant_distance = max(1e-6, lambda_sig * (2.0 * effective_local_scale))
    return DensityScale(
        reference_local_scale=reference_local_scale,
        effective_local_scale=effective_local_scale,
        space_size=space_size,
        significant_distance=significant_distance,
        significant_k=kissing_number(dim),
        scale_factor=scale_factor,
    )
