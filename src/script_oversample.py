import os
import yaml
import pickle
import logging
from pathlib import PurePath
import argparse
import numpy as np
from typing import Tuple

# This adds the BERT path to the python path, needed for the imports inside BERT modules
import sys
sys.path.append('./ml/BERT')

from utils.config_handler import ConfigHandler
from features.data_loader import DataLoader
from ml.ml_pipeline import MLPipeline


def oversamplesimple(settings):
    ch = ConfigHandler(settings)
    ch.get_oversample_experiment_name()

    print(settings["experiment"])

    dl = DataLoader(settings)
    sequences, labels, demographics = dl.load_data()
    xval = MLPipeline(settings)
    xval.train(sequences, labels, demographics)

    config_path = (
        "../experiments/"
        + settings["experiment"]["root_name"]
        + settings["experiment"]["name"]
        + "/config.yaml"
    )
    config_path = PurePath(config_path)
    with open(config_path, "wb") as fp:
        pickle.dump(settings, fp)


def _process_arguments(settings):
    # Oversampling
    if settings["mode"] == "baseline":
        settings["ml"]["oversampler"]["mode"] = "none"
    elif settings["mode"] == "labels":
        settings["ml"]["oversampler"]["mode"] = "ros"
        settings["ml"]["oversampler"]["rebalancing_mode"] = "equal_balancing"
    elif settings["mode"] == "augmentation":
        settings["ml"]["oversampler"]["mode"] = "augmentation"
        settings["ml"]["oversampler"]["rebalancing_mode"] = "equal_balancing"

    oversampling_attributes = "_".join(
        settings["ml"]["oversampler"]["oversampling_col"]
    )
    settings["experiment"]["root_name"] += "/{}_oversampling/{}".format(
        settings["ml"]["oversampler"]["rebalancing_mode"], oversampling_attributes
    )

    settings["experiment"]["labels"] = "binconcepts"
    settings["data"]["others"] = ({"gender": ["3", "4"]},)
    settings["data"]["adjuster"] = {"limit": 819}

    return settings


def main(settings):
    np.random.seed(settings["seeds"]["numpy"])
    settings = _process_arguments(settings)
    oversamplesimple(settings)


if __name__ == "__main__":
    config_path = PurePath("./configs/config.yaml")
    with open(config_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser(description="Oversample")
    # Tasks
    parser.add_argument(
        "--mode",
        dest="mode",
        default=".",
        action="store",
        help="list of the criteria by which to oversample, separated by dots: gender.age",
    )

    settings.update(vars(parser.parse_args()))
    main(settings)


def run_script(settings):
    # run the baseline:
    "$python script_oversample.py --mode baseline"

    # run the model with simple oversampling (rebalancing the labels)
    "$python script_oversample.py --mode labels"

    # run the model with data augmentation (rebalancing the labels, with data augmentation)
    "$python script_oversample.py --mode augmentation"
    ## instructions
    ## 1) Go to src/models/samplers/weier_oversampler.py; In the function oversample:
    ### - Look into the comment blocks ;) There are instruction to create the file
    ### - Either edit directly into the file, or create a new one in src/models/samplers/your_file.py
    ## 2) Go to src/ml/xal_maker.py;
    ### - add your file ml.sampler.your_file to the imports
    ### - in def _choose_sampler(self), add an if statement such that you have:
    ###   if self._pipeline_settings['oversampler'] == 'yourfile' -> chose a keyword you want to use
    ####     self._sampler = YourSampler
    ## 3) Decide how to run your experiment. you can add an if statement in _process_argument (you can set mode via --mode xxx.) Or keep --mode augmentation, and
    ###   change settings['ml']['pipeline']['oversampler'] to your keyword
