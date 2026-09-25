"""
Module to register multiple datasets
"""

import logging
import os
import subprocess

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def register_datasets(path_to_datasets: str):
    """
    Register multiple smartspim datasets using the
    first channel

    Parameters
    -------------
    path_to_datasets: str
        Path to the folder where the datasets
        are located
    """
    datasets = [
        dataset
        for dataset in os.listdir(path_to_datasets)
        if os.path.isdir(f"{path_to_datasets}{dataset}")
    ]
    n_datasets = len(datasets)
    logger.info(f"Datasets: {datasets} - len: {n_datasets}")

    for dataset in datasets:
        cmd = f"python -u main.py --input_data /data/{dataset} --input_channel Ex_488_Em_561 --input_scale 3 --input_zarr_directory processed/OMEZarr"
        logger.info(f"executing {cmd}")
        try:
            for out in execute_command_helper(cmd):
                logger.info(out)
        except ValueError as err:
            logger.error(f"An error occured while executing {dataset} err: {err}")


if __name__ == "__main__":
    path = "/data/"
    register_datasets(path)
