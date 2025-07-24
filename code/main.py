"""
Main used in code ocean to execute capsule
"""

import multiprocessing
import os
import zarr
from ome_zarr.reader import Reader

from aind_ccf_reg import register, utils
from aind_ccf_reg.utils import create_folder, create_logger, read_json_as_dict
from natsort import natsorted
from typing import List, Tuple

def get_zarr_metadata(zarr_path):
    """
    Opens a ZARR file and retrieves its metadata.

    Parameters
    ----------
    zarr_path : str
        file path to zarr file.

    Returns
    -------
    image_node : ome_zarr.reader.Node
        The image node of the ZARR file.
    zarr_meta : dict
        Metadata of the ZARR file.
    """

    zarr_store = zarr.open(zarr_path, mode='r')
    reader = Reader(zarr_store)

    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]
    zarr_meta = image_node.metadata
    return image_node, zarr_meta

def get_estimated_downsample(
    voxel_resolution: List[float], registration_res: Tuple[float] = (16.0, 14.4, 14.4)
) -> int:
    """
    Get the estimated multiscale based on the provided
    voxel resolution. This is used for image stitching.

    e.g., if the original resolution is (1.8. 1.8, 2.0)
    in XYZ order, and you provide (3.6, 3.6, 4.0) as
    image resolution, then the picked resolution will be
    1.

    Parameters
    ----------
    voxel_resolution: List[float]
        Image original resolution. This would be the resolution
        in the multiscale "0".
    registration_res: Tuple[float]
        Approximated resolution that was used for registration
        in the computation of the transforms. Default: (16.0, 14.4, 14.4)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(registration_res[idx] // float(voxel_resolution[idx]))

    downsample_res = int(min(downsample_versions) - 1)
    return downsample_res


def main() -> None:
    """
    Main function to register a dataset
    """
    data_folder = os.path.abspath("../data")
    image_folder = os.path.abspath("../data/fused")
    processing_manifest_path = f"{data_folder}/processing_manifest.json"
    acquisition_path = f"{data_folder}/acquisition.json"

    if not os.path.exists(processing_manifest_path):
        raise ValueError("Processing manifest path does not exist!")

    if not os.path.exists(acquisition_path):
        raise ValueError("Acquisition path does not exist!")

    pipeline_config = read_json_as_dict(processing_manifest_path)
    pipeline_config = pipeline_config.get("pipeline_processing")

    if pipeline_config is None:
        raise ValueError("Please, provide a valid processing manifest")

    acquisition_json = read_json_as_dict(acquisition_path)
    acquisition_orientation = acquisition_json.get("axes")


    if acquisition_orientation is None:
        raise ValueError(
            f"Please, provide a valid acquisition orientation, acquisition: {acquisition_json}"
        )
        
    # Setting parameters based on pipeline
    sorted_channels = natsorted(pipeline_config["registration"]["channels"])

    # Getting highest wavelenght as default for registration
    channel_to_register = sorted_channels[-1]
    additional_channels = pipeline_config["segmentation"]["channels"]

    #calculate downsample for registration
    zarr_path = os.path.join(image_folder, f"{channel_to_register}.zarr/")
    _, acquisition_metadata = get_zarr_metadata(zarr_path)
    acquisition_res = acquisition_metadata['coordinateTransformations'][0][0]['scale'][2:]
    reg_scale = get_estimated_downsample(acquisition_res)
    reg_res = [float(res)/(reg_scale * 1000) for res in acquisition_res]

    results_folder = f"../results/ccf_{channel_to_register}"
    create_folder(results_folder)
    metadata_folder = os.path.abspath(f"{results_folder}/metadata")
    reg_folder = os.path.abspath(f"{metadata_folder}/registration_metadata")
    create_folder(reg_folder)
    create_folder(metadata_folder)

    logger = create_logger(output_log_path=reg_folder)
    logger.info(
        f"Processing manifest {pipeline_config} provided in path {processing_manifest_path}"
    )
    logger.info(f"channel_to_register: {channel_to_register}")

    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info(f"{'='*40} SmartSPIM CCF Registration {'='*40}")

    # ---------------------------------------------------#
    # path to SPIM template, CCF and template-to-CCF registration
    template_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/smartspim_lca_template_25.nii.gz"
    )
    ccf_reference_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/ccf_average_template_25.nii.gz"
    )
    template_to_ccf_transform_warp_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_1Warp.nii.gz"
    )
    template_to_ccf_transform_affine_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_0GenericAffine.mat"
    )
    template_to_ccf_transform_path = [
        template_to_ccf_transform_warp_path,
        template_to_ccf_transform_affine_path,
    ]
    print(f"template_to_ccf_transform_path: {template_to_ccf_transform_path}")

    ccf_to_template_transform_warp_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/syn_1InverseWarp.nii.gz"
    )

    ccf_to_template_transform_path = [
        template_to_ccf_transform_affine_path,
        ccf_to_template_transform_warp_path,
    ]

    print(f"ccf_to_template_transform_path: {ccf_to_template_transform_path}")

    ccf_annotation_to_template_moved_path = os.path.abspath(
        "../data/lightsheet_template_ccf_registration/ccf_annotation_to_template_moved.nii.gz"
    )

    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            "template_path not exist, please provide valid path to SPIM template"
        )

    if not os.path.isfile(ccf_reference_path):
        raise FileNotFoundError(
            "ccf_reference_path not exist, please provide valid path to CCF atlas"
        )

    if not os.path.isfile(template_to_ccf_transform_warp_path):
        raise FileNotFoundError(
            "template_to_ccf_transform_warp_path not exist, please provide valid path"
        )

    if not os.path.isfile(template_to_ccf_transform_affine_path):
        raise FileNotFoundError(
            "template_to_ccf_transform_affine_path not exist, please provide valid path"
        )

    if not os.path.isfile(ccf_annotation_to_template_moved_path):
        raise FileNotFoundError(
            "ccf_annotation_to_template_moved_path not exist, please provide valid path"
        )

    # ---------------------------------------------------#

    regions = read_json_as_dict(
        "../code/aind_ccf_reg/ccf_files/annotation_map.json"
    )
    precompute_path = os.path.abspath("../results/ccf_annotation_precomputed")
    create_folder(precompute_path)
    create_folder(f"{precompute_path}/segment_properties")

    # ---------------------------------------------------#

    example_input = {
        "input_data": image_folder,
        "input_channel": channel_to_register,
        "additional_channels": additional_channels,
        "input_scale": reg_scale,
        "input_orientation": acquisition_orientation,
        "bucket_path": "aind-open-data",
        "template_path": template_path,  # SPIM template
        "ccf_reference_path": ccf_reference_path,
        "template_to_ccf_transform_path": template_to_ccf_transform_path,
        "ccf_to_template_transform_path": ccf_to_template_transform_path,
        "ccf_annotation_to_template_moved_path": ccf_annotation_to_template_moved_path,
        "reference_res": 25,
        "output_data": os.path.abspath(f"{results_folder}/OMEZarr"),
        "metadata_folder": metadata_folder,
        "code_url": "https://github.com/AllenNeuralDynamics/aind-ccf-registration",
        "results_folder": results_folder,
        "reg_folder": reg_folder,
        "prep_params": {
            "rawdata_figpath": f"{reg_folder}/prep_zarr_img.jpg",
            "rawdata_path": f"{reg_folder}/prep_zarr_img.nii.gz",
            "resample_figpath": f"{reg_folder}/prep_resampled_zarr_img.jpg",
            "resample_path": f"{reg_folder}/prep_resampled_zarr_img.nii.gz",
            "mask_figpath": f"{reg_folder}/prep_mask.jpg",
            "mask_path": f"{reg_folder}/prep_mask.nii.gz",
            "n4bias_figpath": f"{reg_folder}/prep_n4bias.jpg",
            "n4bias_path": f"{reg_folder}/prep_n4bias.nii.gz",
            # "img_diff_n4bias_figpath": f"{reg_folder}/prep_img_diff_n4bias.jpg",
            # "img_diff_n4bias_path": f"{reg_folder}/prep_img_diff_n4bias.nii.gz",
            "percNorm_figpath": f"{reg_folder}/prep_percNorm.jpg",
            "percNorm_path": f"{reg_folder}/prep_percNorm.nii.gz",
        },
        "ants_params": {
            "spacing": tuple(reg_res),
            "unit": "millimetre",
            "template_orientations": {
                "anterior_to_posterior": 1,
                "superior_to_inferior": 2,
                "right_to_left": 0,
            },
            "rigid_path": f"{reg_folder}/moved_rigid.nii.gz",
            "affine_path": f"{reg_folder}/moved_affine.nii.gz",
            "moved_to_template_path": f"{reg_folder}/moved_ls_to_template.nii.gz",
            "moved_to_ccf_path": f"{results_folder}/moved_ls_to_ccf.nii.gz",
            "ccf_anno_to_brain_path": f"{reg_folder}/moved_ccf_anno_to_ls.nii.gz",
        },
        "OMEZarr_params": {
            "clevel": 1,
            "compressor": "zstd",
            "chunks": (64, 64, 64),
        },
        "ng_params": {
            "save_path": precompute_path,
            "regions": regions,
            "scale_params": {
                "encoding": "compresso",
                "compressed_block": [16, 16, 16],
                "chunk_size": [32, 32, 32],
                "factors": [2, 2, 2],
                "num_scales": 3,
            },
        },
    }

    logger.info(f"Input parameters in CCF run: {example_input}")
    # flake8: noqa: F841
    image_path = register.main(example_input)

    logger.info(f"Saving outputs to: {image_path}")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_folder,
            "smartspim_ccf_registration",
        )


if __name__ == "__main__":
    main()
