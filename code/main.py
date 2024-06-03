"""
Main used in code ocean to execute capsule
"""

import glob
import json
import logging
import multiprocessing
import os
import subprocess
from datetime import datetime

from aind_ccf_reg import register, utils
from aind_ccf_reg.configs import PathLike
from aind_ccf_reg.utils import create_folder
from natsort import natsorted


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going to be stored

    Returns
    -----------
    logging.Logger
        Created logger
        pointing to the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    LOGS_FILE = f"{output_log_path}/register_process.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    #     logging.disable("DEBUG")
    logging.disable(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Execution datetime: {CURR_DATE_TIME}")

    return logger


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def execute_command_helper(command: str, print_command: bool = False) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    """
    Main function to register a dataset
    """
    data_folder = os.path.abspath("../data/")
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

    results_folder = f"../results/ccf_{channel_to_register}"
    create_folder(results_folder)
    reg_folder = os.path.abspath(f"{results_folder}/registration_metadata")
    metadata_folder = os.path.abspath(f"{results_folder}/metadata")
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

    example_input = {
        "input_data": "../data/fused",
        "input_channel": channel_to_register,
        "input_scale": pipeline_config["registration"]["input_scale"],
        "input_orientation": acquisition_orientation,
        "bucket_path": "aind-open-data",
        "template_path": template_path,  # SPIM template
        "ccf_reference_path": ccf_reference_path,
        "template_to_ccf_transform_path": template_to_ccf_transform_path,
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
            "spacing": (0.0144, 0.0144, 0.016),
            "unit": "millimetre",
            # "ccf_orientations": {
            #     "anterior_to_posterior": 0,
            #     "superior_to_inferior": 1,
            #     "left_to_right": 2,
            # },
            "template_orientations": {
                "anterior_to_posterior": 1,
                "superior_to_inferior": 2,
                "right_to_left": 0,
            },
            "rigid_path": f"{reg_folder}/moved_rigid.nii.gz",
            "moved_to_template_path": f"{reg_folder}/moved_ls_to_template.nii.gz",
            "moved_to_ccf_path": f"{results_folder}/moved_ls_to_ccf.nii.gz",
            "ccf_anno_to_brain_path": f"{reg_folder}/moved_ccf_anno_to_ls.nii.gz",
        },
        "OMEZarr_params": {
            "clevel": 1,
            "compressor": "zstd",
            "chunks": (64, 64, 64),
        },
    }

    logger.info(f"Input parameters in CCF run: {example_input}")
    # flake8: noqa: F841
    image_path = register.main(example_input)

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
