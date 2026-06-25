"""
CZI to Zarr writer. It takes an input path
where 3D stacks are located, then these
stacks are loaded into memory and written
to zarr.
"""

import asyncio
import logging
import multiprocessing
import time
from typing import List, Optional

import numpy as np
import tensorstore as ts

from .omezarr_metadata import (
    _get_pyramid_metadata,
    write_ome_ngff_metadata,
)
import json

def create_spec(
    output_path: str,
    data_shape: list,
    data_dtype: str,
    shard_shape: list,
    chunk_shape: list,
    zyx_resolution: list,
    compressor_kwargs: dict,
    scale: str = "0",
    cpu_cnt: int = None,
    aws_region: str = "us-west-2",
    bucket_name: str = None,
    read_cache_bytes: int = 1 << 30,
) -> dict:
    """
    Create a TensorStore Zarr v3 specification for writing
    to an S3-backed dataset.

    Parameters
    ----------
    output_path : str
        Path inside the S3 bucket where the dataset will be stored.
    data_shape : list of int
        Shape of the full dataset in [t, c, z, y, x] order.
    data_dtype : str
        Data type of the dataset (e.g., 'uint16', 'float32').
    shard_shape : list of int
        Shape of the sharded outer chunks.
    chunk_shape : list of int
        Shape of the internal compressed chunks inside each shard.
    zyx_resolution : list
        Spatial resolution in microns for the z, y, x axes.
    compressor_kwargs: dict
        Compressor parameters for tensorstore
    scale : str, optional
        Scale level identifier (e.g., "0", "1", etc.). Default is "0".
    cpu_cnt : int, optional
        Number of CPU threads to use. Defaults to the number of system CPUs.
    aws_region : str, optional
        AWS region where the S3 bucket resides. Default is "us-west-2".
    bucket_name : str, optional
        Name of the S3 bucket. Default is "aind-msma-morphology-data".
    read_cache_bytes : int, optional
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    spec : dict
        TensorStore specification dictionary to create a new Zarr v3 dataset.
    """
    output_path = str(output_path)

    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    zyx_resolution = [
        f"{r}um" if r is not None else None for r in zyx_resolution
    ]
    zyx_resolution = [None] * (5 - len(zyx_resolution)) + zyx_resolution

    kvstore_dict = {
        "driver": "file",
    }

    if bucket_name is not None:
        kvstore_dict = {
            "driver": "s3",
            "bucket": bucket_name,
            "aws_region": aws_region,
            # "aws_credentials": {
            #     "type": "profile",
            #     "profile": aws_profile,
            #     "credentials_file": os.path.expanduser(credentials_file),
            # },
            "context": {
                "cache_pool": {"total_bytes_limit": read_cache_bytes},
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
                "experimental_s3_rate_limiter": {
                    "read_rate": cpu_cnt,
                    "write_rate": cpu_cnt,
                },
            },
        }

    return {
        "driver": "zarr3",
        "kvstore": {
            **kvstore_dict,
            "path": output_path,
        },
        "path": str(scale),
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "metadata": {
            "shape": data_shape,
            "zarr_format": 3,
            "node_type": "array",
            "fill_value": 0,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shard_shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "attributes": {
                "dimension_units": zyx_resolution,
            },
            "dimension_names": ["t", "c", "z", "y", "x"],
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {
                                "name": "blosc",
                                "configuration": compressor_kwargs,
                            },
                        ],
                        "index_codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {"name": "crc32c"},
                        ],
                        "index_location": "end",
                    },
                }
            ],
            "data_type": data_dtype,
        },
        "create": True,
        "delete_existing": True,
    }


async def create_downsample_dataset(
    dataset_path: str,
    start_scale: int,
    downsample_factor: list,
    downsample_mode: str,
    compressor_kwargs: dict,
    cpu_cnt: int = None,
    aws_region: str = "us-west-2",
    bucket_name: str = None,
    read_cache_bytes: int = 1 << 30,
):
    """
    Create a new downsampled scale level in a multi-scale Zarr v3 dataset.

    Parameters
    ----------
    dataset_path : str
        Base S3 path of the dataset inside the bucket.
    start_scale : int
        The scale level to downsample from (e.g., 0 for original resolution).
    downsample_factor : list of int
        Downsampling factor for each of [t, c, z, y, x] dimensions.
    downsample_mode : str
        Downsampling method. Options are: stride, median, mode, mean, min, max.
    compressor_kwargs: Dict
        Blosc compressor arguments for tensorstore
    cpu_cnt : int, optional
        Number of threads to use. If None, uses all available CPUs.
    aws_region : str, optional
        AWS region of the S3 bucket.
    bucket_name : str, optional
        S3 bucket name.
    read_cache_bytes : int, optional
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    None
        The function writes a new downsampled scale directly
        to the same Zarr dataset.
    """
    dataset_path = str(dataset_path)

    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    kvstore_dict = {
        "driver": "file",
    }

    if bucket_name is not None:
        kvstore_dict = {
            "driver": "s3",
            "bucket": bucket_name,
            "aws_region": aws_region,
            "context": {
                "cache_pool": {"total_bytes_limit": read_cache_bytes},
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
                "experimental_s3_rate_limiter": {
                    "read_rate": cpu_cnt,
                    "write_rate": cpu_cnt,
                },
            },
        }

    downsample_factor = [1] * (5 - len(downsample_factor)) + downsample_factor

    source_w_down_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factor,
        "downsample_method": downsample_mode,
        "base": {
            "driver": "zarr3",
            "kvstore": {
                **kvstore_dict,
                "path": dataset_path,
            },
            "path": str(start_scale),
            "recheck_cached_metadata": False,
            "recheck_cached_data": False,
        },
    }

    downsampled_dataset = await ts.open(spec=source_w_down_spec)
    source_dataset = downsampled_dataset.base
    new_scale = start_scale + 1

    downsampled_resolution = [
        unit.multiplier if isinstance(unit, ts.Unit) else unit
        for unit in downsampled_dataset.dimension_units
    ]

    # Creates downsample spec
    # Keeping chunk_size equal in all dimensions, however it might be
    # suboptimal? We might want to have chunks of ~50-100 MB
    down_spec = create_spec(
        output_path=dataset_path,
        data_shape=downsampled_dataset.shape,
        data_dtype=source_dataset.dtype.name,
        shard_shape=source_dataset.chunk_layout.write_chunk.shape,
        chunk_shape=source_dataset.chunk_layout.read_chunk.shape,
        zyx_resolution=downsampled_resolution,
        scale=new_scale,
        cpu_cnt=cpu_cnt,
        aws_region=aws_region,
        bucket_name=bucket_name,
        compressor_kwargs=compressor_kwargs,
    )

    down_dataset = await ts.open(down_spec)
    downsampled_data = await downsampled_dataset.read()
    await down_dataset.write(downsampled_data)


def zarr_writer(
    image_data: np.ndarray,
    output_path: str,
    voxel_size: List[float],
    shard_size: List[int],
    chunk_size: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    logger: logging.Logger,
    stack_name: str,
    compressor_kwargs: dict,
    downsample_mode: Optional[str] = "mean",
    bucket_name: Optional[str] = None,
):
    """
    Writes a fused Zeiss channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    image_data: np.ndarray
        Numpy array containing the image data.

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    chunk_size: List[int]
        Final chunk_size we want to use to write
        the final dataset

    codec: str
        Image codec for writing the Zarr

    compression_level: int
        Compression level

    scale_factor: List[int]
        Scale factor per axis. The dimensionality
        is organized as ZYX.

    n_lvls: int
        Number of levels on the pyramid (multiresolution)
        for better visualization

    channel_name: str
        Channel name we are currently writing

    logger: logging.Logger
        Logger object

    stack_name: str
        Stack name for the zarr3

    compressor_kwargs: Dict
        Blosc compressor arguments for tensorstore

    downsample_mode: str
        Downsample mode for the dataset.
        Options are: stride, median, mode, mean, min, max
        Default is mean.

    bucket_name: Optional[str] = None
        Bucket name to upload the dataset.
        If it is None, then it will be stored locally.
    """
    output_path = f"{output_path}/{stack_name}"
    start_time = time.time()
    
    dataset_shape = tuple(i for i in image_data.shape if i != 1)
    extra_axes = (1,) * (5 - len(dataset_shape))
    dataset_shape = extra_axes + dataset_shape

    shard_size = ([1] * (5 - len(shard_size))) + list(shard_size)
    chunk_size = ([1] * (5 - len(chunk_size))) + list(chunk_size)

    # Getting channel color
    channel_colors = None

    logger.info(
        f"Writing from {stack_name} to {output_path} bucket {bucket_name}"
    )

    if np.issubdtype(image_data.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(image_data.dtype).min,
            np_info_func(image_data.dtype).max,
        )
        for _ in range(dataset_shape[1])
    ]

    vmin, vmax = np.percentile(image_data, (0.1, 99.9))

    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(vmin, vmax) for _ in range(dataset_shape[1])]

    # Writing OME-NGFF metadata
    multiscale_zarr_json = write_ome_ngff_metadata(
        arr_shape=dataset_shape,
        image_name=stack_name,
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_names=[channel_name],
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=_get_pyramid_metadata(),
        chunk_size=chunk_size,
        origin=[0, 0, 0],  # TODO get nominal coordinates into metadata
    )

    # Full resolution spec
    spec = create_spec(
        output_path=output_path,
        bucket_name=bucket_name,
        data_shape=dataset_shape,
        data_dtype=image_data.dtype.name,
        shard_shape=shard_size,
        chunk_shape=chunk_size,
        zyx_resolution=voxel_size,
        compressor_kwargs=compressor_kwargs,
    )

    dataset = ts.open(spec).result()

    # Write full resolution data
    dataset.write(image_data).result()

    # Base scale "0" is already written above, so we only create n_lvls - 1
    # additional downsampled levels to match the n_lvls datasets declared in
    # the OME-NGFF metadata.
    for level in range(n_lvls - 1):
        asyncio.run(
            create_downsample_dataset(
                dataset_path=output_path,
                start_scale=level,
                downsample_factor=scale_factor,
                downsample_mode=downsample_mode,
                compressor_kwargs=compressor_kwargs,
                bucket_name=bucket_name,
            )
        )

    # Writes top level json
    with open(f"{output_path}/zarr.json", "w") as fp:
        json.dump(multiscale_zarr_json, fp, indent=2)

    end_time = time.time()
    logger.info(f"Time to write the dataset: {end_time - start_time}")
