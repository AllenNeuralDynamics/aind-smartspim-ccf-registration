"""
Main used in code ocean to execute capsule
"""

import math
import multiprocessing
import os
from typing import List, Tuple

import zarr
from aind_ccf_reg import register, utils
from aind_ccf_reg.utils import create_folder, create_logger, read_json_as_dict
from natsort import natsorted
from ome_zarr.reader import Reader

import argparse
import sys

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

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # Open zarr group using the path directly
    zarr_group = zarr.open(zarr_path, mode="r")

    # Ensure we have a Group object (not Array)
    if not isinstance(zarr_group, zarr.Group):
        raise ValueError(f"Expected zarr Group, got {type(zarr_group)}")

    # Add the exists method that ome-zarr Reader expects
    if not hasattr(zarr_group, "exists"):
        zarr_group.exists = lambda: True

    reader = Reader(zarr_group)

    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]
    zarr_meta = image_node.metadata
    return image_node, zarr_meta


def get_estimated_downsample(
    voxel_resolution: List[float],
    registration_res: Tuple[float, float, float] = (16.0, 14.4, 14.4),
) -> int:
    """
    Estimate the multiscale downsample level based on voxel resolution.

    Example:
    --------
    If voxel_resolution = (1.8, 1.8, 2.0) and registration_res = (3.6, 3.6, 4.0),
    the result will be 1, because the registration resolution is about 2× coarser.

    Parameters
    ----------
    voxel_resolution : List[float]
        Resolution of the image at scale 0 (e.g., in microns per voxel).
    registration_res : Tuple[float, float, float]
        Approximate resolution used for registration.

    Returns
    -------
    int
        Estimated downsample level.
    """
    ratios = [
        registration_res[i] / float(voxel_resolution[i])
        for i in range(len(voxel_resolution))
    ]

    # Choose the smallest ratio across dimensions (safest valid downsample factor)
    downsample_factor = min(ratios)

    # Convert to nearest integer downsample level
    return int(round(math.log2(downsample_factor)))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stitched',
        default = "SmartSPIM_693196_2023-09-28_23-12-22_stitched_2024-01-11_10-23-15",
        type = str,
        help = 'Name of stitched data base dir (e.g. SmartSPIM_693196_2023-09-28_23-12-22_stitched_2024-01-11_10-23-15)')

    registration_channel = parser.add_argument('--channel',
        default = 'Ex_639_Em_660',
        type = str,
        help = 'Channel to register')
    
    force_180 = parser.add_argument('--rotate',
        default= True,
        type = str2bool,
        help = 'Should we rotate 180? Use true only for bad metadata')
    
    additional_channels = parser.add_argument('--extra',
        default= 'Ex_561_Em_593,Ex_488_Em_525',
        type = str,
        help = 'Should we rotate 180? Use true only for bad metadata')
    
    return parser


def main() -> None:
    """
    Main function to register a dataset
    """
    parser = get_parser()
    args = parser.parse_args()

    spim_image_stitched = args.stitched
    registration_channel = args.channel
    additional_channels = args.extra.split(',')
    

    data_folder = os.path.abspath(f"../data")
    results_path = os.path.abspath(f"../results")
    image_folder = os.path.abspath(f"../data/{spim_image_stitched}")

    # processing_manifest_path = f"{data_folder}/processing.json"

    acquisition_path = f"{image_folder}/acquisition.json"

    # print(processing_manifest_path)
    # if not os.path.exists(processing_manifest_path):
    #     raise ValueError("Processing manifest path does not exist!")
    
    print(acquisition_path)

    if not os.path.exists(acquisition_path):
        raise ValueError("Acquisition path does not exist!")

    # pipeline_config = read_json_as_dict(processing_manifest_path)
    # pipeline_config = pipeline_config.get("pipeline_processing")

    # if pipeline_config is None:
    #     raise ValueError("Please, provide a valid processing manifest")

    # registration_info = pipeline_config.get("registration")

    # if registration_info is None:
    #     raise ValueError("Please, provide registration channels.")

    #channels_to_process = registration_info.get("channels")

    acquisition_json = read_json_as_dict(acquisition_path)
    acquisition_orientation = acquisition_json.get("axes")

    if acquisition_orientation is None:
        raise ValueError(
            f"Please, provide a valid acquisition orientation, acquisition: {acquisition_json}"
        )
    
    print(acquisition_orientation)
    if args.rotate:
        print('DOING ROTATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        LR = False
        AP = False
        for ii,ax in enumerate(acquisition_orientation):
            if (not LR) and (ax['direction']=="Left_to_right"):
                ax['direction']="Right_to_left"
                LR = True
            elif (not LR) and (ax['direction']=="Right_to_left"):
                ax['direction']="Left_to_right"
                LR = True
            elif (not AP) and (ax['direction']=="Posterior_to_anterior"):
                ax['direction']="Anterior_to_posterior"
                AP = True
            elif (not AP) and (ax['direction']=="Anterior_to_posterior"):
                ax['direction']="Posterior_to_anterior"
                AP = True
        print(acquisition_orientation)

    else:
        print('Not doing rotation')
    
    # # Setting parameters based on pipeline
    # sorted_channels = natsorted(
    #     pipeline_config["registration"]["channels"]
    # )

    # # Getting highest wavelenght as default for registration
    # channel_to_register = sorted_channels[-1]
    # additional_channels = pipeline_config["segmentation"]["channels"]
    channel_to_register = registration_channel
    additional_channels = additional_channels # 

    # Create output folders
    results_folder = f"../results/ccf_{channel_to_register}"
    create_folder(results_folder)
    metadata_folder = os.path.abspath(f"{results_folder}/metadata")
    reg_folder = os.path.abspath(
        f"{metadata_folder}/registration_metadata"
    )
    create_folder(reg_folder)
    create_folder(metadata_folder)

    logger = create_logger(output_log_path=reg_folder)

    # Calculate downsample for registration
    zarr_attrs_path = os.path.join(
        image_folder,'image_tile_fusing','OMEZarr', f"{channel_to_register}.zarr/.zattrs"
    )
    acquisition_metadata = utils.read_json_as_dict(zarr_attrs_path)
    acquisition_res = acquisition_metadata["multiscales"][0]["datasets"][
        0
    ]["coordinateTransformations"][0]["scale"][2:]
    logger.info(
        f"Image was acquired at resolution (um): {acquisition_res}"
    )
    reg_scale = get_estimated_downsample(acquisition_res)
    logger.info(f"Image is being downsampled by a factor: {reg_scale}")
    reg_res = [
        (float(res) * 2**reg_scale) / 1000 for res in acquisition_res
    ]
    logger.info(f"Registration resolution (mm): {reg_res}")

    # logger.info(
    #     f"Processing manifest {pipeline_config} provided in path {processing_manifest_path}"
    # )
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
        f"{data_folder}/lightsheet_template_ccf_registration/smartspim_lca_template_25.nii.gz"
    )
    ccf_reference_path = os.path.abspath(
        f"{data_folder}/allen_mouse_ccf/average_template/average_template_25.nii.gz"
    )
    ccf_annotation_path = os.path.abspath(
        f"{data_folder}/allen_mouse_ccf/annoation/ccf_2017/annoation_25.nii.gs)

    template_to_ccf_transform_warp_path = os.path.abspath(
        f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_1Warp.nii.gz"
    )
    template_to_ccf_transform_affine_path = os.path.abspath(
        f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_0GenericAffine.mat"
    )
    template_to_ccf_transform_path = [
        template_to_ccf_transform_warp_path,
        template_to_ccf_transform_affine_path,
    ]
    print(
        f"template_to_ccf_transform_path: {template_to_ccf_transform_path}"
    )

    ccf_to_template_transform_warp_path = os.path.abspath(
        f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_1InverseWarp_25.nii.gz"
    )

    ccf_to_template_transform_path = [
        template_to_ccf_transform_affine_path,
        ccf_to_template_transform_warp_path,
    ]

    print(
        f"ccf_to_template_transform_path: {ccf_to_template_transform_path}"
    )

    ccf_annotation_to_template_moved_path = os.path.abspath(
        f"{data_folder}/lightsheet_template_ccf_registration/ccf_annotation_to_template_moved.nii.gz"
    )

    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            f"template_path {template_path} not exist, please provide valid path to SPIM template"
        )

    if not os.path.isfile(ccf_reference_path):
        raise FileNotFoundError(
            f"ccf_reference_path {ccf_reference_path} not exist, please provide valid path to CCF atlas"
        )

    if not os.path.isfile(template_to_ccf_transform_warp_path):
        raise FileNotFoundError(
            f"template_to_ccf_transform_warp_path {template_to_ccf_transform_warp_path} not exist, please provide valid path"
        )

    if not os.path.isfile(template_to_ccf_transform_affine_path):
        raise FileNotFoundError(
            f"template_to_ccf_transform_affine_path {template_to_ccf_transform_affine_path} not exist, please provide valid path"
        )

    if not os.path.isfile(ccf_annotation_to_template_moved_path):
        raise FileNotFoundError(
            f"ccf_annotation_to_template_moved_path {ccf_annotation_to_template_moved_path} not exist, please provide valid path"
        )

    # ---------------------------------------------------#

    regions = read_json_as_dict(
        "../code/aind_ccf_reg/ccf_files/annotation_map.json"
    )
    precompute_path = os.path.abspath(
        "../results/ccf_annotation_precomputed"
    )
    create_folder(precompute_path)
    create_folder(f"{precompute_path}/segment_properties")

    # ---------------------------------------------------#

    example_input = {
        "input_data": os.path.join(image_folder,'image_tile_fusing','OMEZarr'),
        "input_channel": channel_to_register,
        "additional_channels": additional_channels,
        "input_scale": reg_scale,
        "input_orientation": acquisition_orientation,
        "bucket_path": "aind-open-data",
        "template_path": template_path,  # SPIM template
        "ccf_reference_path": ccf_reference_path,
        "ccf_annotation_path":ccf_annotation_path,
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
