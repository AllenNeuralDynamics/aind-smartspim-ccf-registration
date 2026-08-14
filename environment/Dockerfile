FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# No git credential helper. The only dependency fetched over git is
# aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git, a public
# repository -- verified to clone with no helper present at all. The previous
# `ARG GIT_ASKPASS` / `COPY git-askpass /` existed to authenticate it and only
# ever added a way for the build to fail: Code Ocean's materialised filename has
# moved between `git-askpass` and `git-ask-pass`, and a COPY of the wrong one
# aborts the build before a single dependency is fetched. Nothing here needs it.

# The env is python 3.10, not 3.8. Nothing here needs 3.10 as such -- the pins
# below are pure python or ship wheels either way. The problem is four C
# extensions pulled in transitively by cloud-volume[all_codecs] (zfpy,
# pysimdjson, dracopy, multiprocess) that have stopped publishing cp38 wheels.
# They are not pinned, so pip takes the newest, finds no wheel, and builds from
# source; zfpy's Cython 2 syntax then fails under the Cython 3 that build
# isolation installs:
#
#     python/zfpy.pyx:93:28: 'ndarray' is not a type identifier
#
# So this was never a code problem -- the 16 pins below were fine, and the ~150
# unpinned transitives aged out from under them. On 3.10 all four have wheels
# and the install is pure download.
#
# The ceiling is 3.10: scikit-image 0.19.3 publishes no wheels past cp310 and
# xarray-multiscale 1.1.0 requires <3.11, so the usable window is 3.9-3.10.
#
# The env's python is independent of the base image's, which is why this is a
# one-token change and the base image stays put. code/run does
# `conda activate ccf_reg`, so this env is what actually executes.

RUN conda create -n ccf_reg python=3.10

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ccf_reg", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    antspyx==0.4.2 \
    argschema==3.0.4 \
    s3fs==2022.11.0 \
    scikit-image==0.19.3 \
    tifffile==2022.10.10 \
    bokeh==2.4.2 \
    aind-data-schema==1.0.0 \
    xarray_multiscale==1.1.0 \
    dask[distributed]==2022.11.1 \
    cloud-volume[all_codecs]==11.0.2 \
    matplotlib==3.7.3 \
    ome-zarr==0.8.2 \
    natsort==8.4.0 \
    crackle-codec==0.26.0 \
    aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs \
    zarr==2.16.1 \
    awscli
