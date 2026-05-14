# aind-ccf-registration

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

Source code to register SmartSPIM lightsheet datasets to the Allen Common Coordinate Framework (CCF) atlas. This module is part of a larger pipeline and is compatible with the Code Ocean Pipeline feature.

The pipeline assumes the fused image is provided in OME-Zarr format with multiple resolution levels. Because ANTs cannot operate at native resolution for large lightsheet volumes, the pipeline uses a downsampled multiscale level (estimated automatically from the acquisition metadata). The image is then resampled to match the CCF reference resolution before registration.

$$resX=origResX*(2^m)$$
$$resY=origResY*(2^m)$$
$$resZ=origResZ*(2^m)$$

Where $m$ is the estimated multiscale level. Afterwards, we resample the image to the target resolution (default 25 µm) and register it to the Allen CCF atlas.

The SmartSPIM lightsheet template is located in our [public S3 bucket](https://open.quiltdata.com/b/aind-open-data/tree/SmartSPIM-template_2024-05-16_11-26-14/). Please check the readme in that path for information about the template and the required CCF reference files.

---

## Pipeline Overview

The registration pipeline executes these stages in order:

1. **Orientation check** — Reorients the input image axes to match the SPIM template coordinate system.
2. **Preprocessing** — Applies brain masking, percentile intensity normalization, resampling to the registration resolution, and N4 bias field correction.
3. **Rigid registration** — Coarse alignment to the SPIM template using rigid body transforms (ANTs).
4. **Affine registration** — Linear alignment refining scale and shear (ANTs).
5. **SyN registration** — Deformable alignment using symmetric diffeomorphic normalization (ANTs).
6. **Template → CCF mapping** — Applies pre-computed template-to-CCF transforms to map the registered image into CCF space.
7. **Reverse transforms** — Maps the CCF annotation volume back into the original brain space; generates Neuroglancer precompute segmentation layers.
8. **Additional channel alignment** — Applies the computed transforms to any auxiliary imaging channels.
9. **OME-Zarr output** — Writes the registered volume with multi-scale pyramids using Dask distributed.

---

## Input / Output Formats

### Input

- **Fused image**: OME-Zarr directory with at least one channel subdirectory (e.g., `Ex_488_Em_561.zarr`). The `.zattrs` file must contain `multiscales` metadata with voxel resolution.
- **Processing manifest** (`processing_manifest.json`): JSON file specifying which channels to register and segment.
- **Acquisition metadata** (`acquisition.json`): JSON file with `axes` field describing the acquisition orientation.
- **SPIM template**: NIfTI image (`.nii.gz`) aligned to the lightsheet coordinate system.
- **CCF reference & annotation**: NIfTI images from the Allen CCF atlas at 10 µm or 25 µm.
- **Template-to-CCF transforms**: ANTs transform files (`.mat` and/or displacement fields) mapping the SPIM template to CCF space.

### Output

| Output | Format | Description |
|--------|--------|-------------|
| Registered brain | OME-Zarr (multi-scale) | Brain volume in CCF space |
| Registration intermediates | NIfTI (`.nii.gz`) | Rigid, affine, SyN moved images |
| CCF annotation in brain space | NIfTI + Neuroglancer precompute | Atlas annotations mapped back |
| QC figures | PNG | 3-plane overlays at each registration stage |
| Processing metadata | JSON | Provenance records for each processing step |
| Resource usage graphs | PNG | CPU/memory profile of the run |

---

## System Requirements

- **CPU**: Multi-core system recommended; Dask distributed scales workers to available CPUs.
- **Memory**: ≥32 GB RAM recommended for 25 µm registration; ≥64 GB for 10 µm.
- **Disk**: Sufficient space for intermediate NIfTI files (~1–5 GB per stage) plus OME-Zarr output.
- **Key dependencies**: `antspyx`, `dask[distributed]`, `zarr`, `ome-zarr`, `scikit-image`, `aind-data-schema`.
- **CCF template files**: Must be downloaded separately from the AIND public S3 bucket (see link above).

---

## Installation

To use the software, in the root directory, run:

```bash
pip install -e .
```

To develop the code, run:

```bash
pip install -e .[dev]
```

---

## Usage

The pipeline is designed to run inside a Code Ocean capsule where data is mounted at `../data` and results are written to `../results`. The entry point reads `../data/processing_manifest.json` and `../data/acquisition.json` automatically.

```bash
python code/main.py
```

The processing manifest must contain a `pipeline_processing` section specifying registration and segmentation channels:

```json
{
  "pipeline_processing": {
    "registration": {
      "channels": ["Ex_647_Em_690"]
    },
    "segmentation": {
      "channels": ["Ex_488_Em_561"]
    }
  }
}
```

To register multiple datasets in batch (development use):

```bash
python code/register_datasets.py
```

---

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```
sphinx-apidoc -o doc_template/source/ code
```
Then to create the documentation html files, run
```
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found here: https://www.sphinx-doc.org/en/master/usage/installation.html
