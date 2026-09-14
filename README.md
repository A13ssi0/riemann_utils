# riemann_utils

Utilities for working with symmetric positive definite covariance matrices, Riemannian geometry, Laplacian masks, and visualization helpers for EEG/BCI analysis.

This repository can be used independently from any parent project. It is mainly intended for scripts that work with covariance matrices shaped by frequency band, run/session, class, and channel.

## Modules

```text
.
|-- covariances.py        # Riemannian means, covariance centering, PCA projection, rank helpers
|-- matrix_functions.py   # Riemannian metrics, angles, distances, and run-to-run matrices
|-- laplacian.py          # EEG channel locations and Laplacian mask generation
|-- plots.py              # UMAP, centroid, polar, and Cartesian plotting helpers
`-- LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/A13ssi0/riemann_utils.git
```

Then import it from a script that can see the repository folder:

```python
from riemann_utils.covariances import get_riemann_mean_covariance
from riemann_utils.matrix_functions import compute_runMatrix_angles
```

If you use this repository as a Git submodule, make sure the parent project adds the parent directory to `sys.path`, or run scripts from a location where `riemann_utils` is importable.

## Dependencies

Install the common dependencies with:

```bash
python -m pip install numpy scipy matplotlib scikit-learn pyriemann umap-learn colour
```

Depending on the functions you use, not every dependency is always required. For example, Riemannian covariance helpers require `pyriemann`, while UMAP plots require `umap-learn`.

## Covariance Utilities

`covariances.py` contains helpers for covariance matrices and low-dimensional projections.

Common functions:

- `get_riemann_mean_covariance()`: compute the Riemannian mean covariance for each frequency band.
- `center_covariances()`: center covariance matrices around a reference matrix.
- `center_covariance_online()`: online version of covariance centering.
- `get_nd_position()`: project data to a lower-dimensional PCA space.
- `matrix_to_maxRank()` and `getrank()`: inspect or work around rank-deficient data.
- `get_trials_gradient()`: create a trial-progress vector from sample/window indices.
- `colorFader()`: interpolate between two colors.
- `get_canonical_bases()`: return canonical basis vectors.

Example:

```python
from riemann_utils.covariances import (
    get_riemann_mean_covariance,
    center_covariances,
)

mean_cov, feedback_idx = get_riemann_mean_covariance(covs)
centered_covs = center_covariances(covs, mean_cov)
```

Expected covariance shape for most functions:

```text
bands x samples/windows x channels x channels
```

## Matrix Geometry

`matrix_functions.py` contains helpers for comparing SPD matrices on the Riemannian manifold.

Common functions:

- `metric_riemann()`: compute the Riemannian inner product at a tangent point.
- `norm_riemann()`: compute the Riemannian norm.
- `angle_between_matrices()`: compute angle and cosine similarity between two SPD matrices.
- `compute_runMatrix_angles()`: compute run-to-run angle matrices.
- `compute_runMatrix_distances()`: compute run-to-run normalized Riemannian distance matrices.
- `matrix_std()`: compute matrix dispersion around a center point.
- `matrix_meanAbsoluteDeviation()`: compute mean absolute Riemannian deviation.
- `evaluate_negative_angles()`: assign signs to angles according to a reference direction.

Example:

```python
from riemann_utils.matrix_functions import compute_runMatrix_distances

distance_matrix = compute_runMatrix_distances(
    run_centroids=centroids,
    mAbsDev_centroids=centroid_deviation,
    doPrint=False
)
```

Run-centroid helpers accept data shaped as:

```text
bands x runs x classes x channels x channels
```

If the class dimension is missing, the current helpers add a singleton class dimension automatically.

## Laplacian Utilities

`laplacian.py` contains standard EEG channel locations and utilities to build spatial Laplacian masks.

Common functions/classes:

- `EEG_CHANNEL`: small container for channel label and 3D/spherical coordinates.
- `get_standard_chanlocs()`: return built-in standard EEG channel locations.
- `get_laplacianMask()`: generate classic and distance-weighted Laplacian masks.
- `plot_lapMask()`: visualize Laplacian channel connections in 3D.
- `cart2sph()`: convert Cartesian coordinates to spherical coordinates.

Example:

```python
from riemann_utils.laplacian import get_laplacianMask

channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
lap_mask, weighted_lap_mask, chanlocs = get_laplacianMask(
    channels=channels,
    distance=4,
    isDistanceMeasure=False
)
```

Note: `apply_laplacian()` currently uses hard-coded local file paths for precomputed masks. For reusable code, prefer `get_laplacianMask()` or adapt `apply_laplacian()` to load masks from your own project paths.

## Plotting Utilities

`plots.py` contains visualization helpers for reduced-dimensional covariance/centroid data.

Common functions:

- `umap_reduction()`: compute UMAP embeddings by band.
- `plot_cartesian()`: plot 2D or 3D points colored by class/session/trial progress.
- `polarPlot_centroids()`: plot centroid trajectories in polar coordinates.
- `plot_centroids_movement()`: plot centroid distance or movement over runs.
- `plot_centroids_angles()`: plot centroid-angle matrices.
- `generate_equally_distributed_colors()` and `get_color_gradient()`: color utilities.
- `nd_to_3d_polar()` and `traslate_eye_to_zero()`: coordinate transformations.

Example:

```python
from riemann_utils.plots import umap_reduction, plot_cartesian

embedding, reducer = umap_reduction(data, n_components=3)
plot_cartesian(embedding[0], labels=labels)
```

## Notes

- Most functions assume input matrices are symmetric positive definite.
- Shape conventions are important; check each function before mixing sample/run/class dimensions.
- Several utilities were written for EEG/BCI experiments, but the matrix-geometry functions can also be useful for other SPD-matrix workflows.
- This repository is a utility collection rather than a fully packaged Python distribution; import paths may need to be configured by the calling project.

## License

This project is released under the MIT License. See `LICENSE` for details.
