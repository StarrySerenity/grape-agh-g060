"""Fixed, interpretable descriptors of estimated berry-center geometry."""
import numpy as np
from scipy.spatial.distance import cdist, pdist


def describe(models: list[dict]) -> dict:
    if not isinstance(models, list) or len(models) < 4:
        raise ValueError('At least four estimated berries are required')
    try:
        centers = np.asarray([m['center'] for m in models], dtype=float)
        radii = np.asarray([m['radius'] for m in models], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid berry geometry fields') from exc
    if centers.shape != (len(models), 3) or not np.isfinite(centers).all():
        raise ValueError('Centers must be finite XYZ triples')
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Radii must be finite and positive')
    unique = len(np.unique(centers, axis=0))
    if unique < 4:
        raise ValueError('At least four distinct centers are required')
    pair_distances = pdist(centers)
    if not np.isfinite(pair_distances).all():
        raise ValueError('Coordinate magnitude prevents finite distance computation')
    scale = float(np.median(pair_distances[pair_distances > 0]))
    normalized = (centers - centers.mean(axis=0)) / scale
    eig = np.maximum(np.linalg.eigvalsh(normalized.T @ normalized / len(models))[::-1], 0)
    eigen_ratios = eig / eig.sum()
    median_radius = float(np.median(radii))
    distances = cdist(centers, centers)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    pairs = np.triu_indices(len(models), 1)
    overlap = float(np.mean(pair_distances < (radii[:, None] + radii[None, :])[pairs]))
    radial = np.linalg.norm(normalized, axis=1)
    scalar = np.array([np.log1p(len(models)) / np.log1p(200), *eigen_ratios,
                       median_radius / scale, np.std(radii) / median_radius,
                       np.median(nearest) / scale, overlap])

    def histogram(values, bins, high):
        counts = np.histogram(np.clip(values, 0, high), bins=bins, range=(0, high))[0]
        return np.sqrt(counts / counts.sum())

    blocks = [scalar, histogram(pair_distances / scale, 24, 3),
              histogram(radial, 16, 3), histogram(radii / median_radius, 12, 3),
              histogram(nearest / median_radius, 12, 6)]
    vector = np.concatenate([b / np.sqrt(len(b)) for b in blocks])
    if not np.isfinite(vector).all():
        raise ValueError('Geometry normalization produced nonfinite values')
    return dict(vector=vector.tolist(), baseline=(scalar / np.sqrt(len(scalar))).tolist(),
                metrics=dict(berry_count=len(models), distinct_centers=unique,
                             median_radius_over_center_scale=median_radius / scale,
                             covariance_minor_major_ratio=float(eig[-1] / eig[0]),
                             covariance_middle_major_ratio=float(eig[1] / eig[0]),
                             radius_coefficient_variation=float(np.std(radii) / median_radius),
                             nearest_neighbor_over_scale=float(np.median(nearest) / scale),
                             estimated_sphere_overlap_fraction=overlap),
                warnings=(['duplicate centers present'] if unique != len(models) else []) +
                         ['estimated spheres; not individually verified ground truth',
                          'dimensionless geometry; no physical growth rate'])
