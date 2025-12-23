FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n ccf_reg python=3.12.4

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ccf_reg", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    antspyx==0.6.1 \
    argschema==3.0.4 \
    s3fs==2023.6 \
    scikit-image==0.24 \
    tifffile==2022.10.10 \
    bokeh==3.0 \
    aind-data-schema==1.0.0 \
    xarray_multiscale==1.2.0 \
    dask[array]==2024.3.0 \
    distributed==2024.3.0 \
    cloud-volume[all_codecs]==11.0.2 \
    matplotlib==3.10.8 \
    ome-zarr==0.8.2 \
    natsort==8.4.0 \
    crackle-codec==0.26.0 \
    aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs \
    zarr==2.17 \
    numcodecs==0.12.1\
    awscli
    
RUN pip install -U --no-cache-dir\
    aind-zarr-utils==0.13.2
