# Data format

The Python experiment scripts read MATLAB MAT files through `scipy.io.loadmat`.
Each dataset must include the following arrays:

| Variable | Shape | Unit | Description |
| --- | --- | --- | --- |
| `velocity2` | `N x 256` | task-specific dispersion quantity | Input dipole acoustic dispersion response sampled at 256 points. |
| `param2` | `N x 10` | m/s | Target radial shear-wave velocity profile from the borehole wall outward. |

The 10 values in each row of `param2` represent equal radial layers. In the
manuscript visualization convention, the layer centers are 0.15, 0.25, ..., and
1.05 m, corresponding to a radial interval of 0.1-1.1 m.

For structured-output experiments, the target profile must satisfy the physical
prior selected for the experiment. The default paper configuration constrains
velocity to 1200-3500 m/s and maps the model output to a monotonic radial
recovery profile.

The repository includes `examples/training_subset_10000.mat`, a real
10,000-sample training subset, and `python examples/quick_test.py` creates a
small synthetic MAT file for format verification. The full training dataset is
released separately through Zenodo; add the assigned DOI URL here after
publication. Field data are not redistributed.
