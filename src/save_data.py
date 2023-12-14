import os
import yaml
import pickle
import logging
from pathlib import PurePath
import argparse
import numpy as np
from typing import Tuple

import sys
sys.path.append('./ml/BERT')
from utils.config_handler import ConfigHandler
from features.data_loader_test_for_full_data import DataLoader
from ml.ml_pipeline import MLPipeline


def oversamplesimple(settings):
    ch = ConfigHandler(settings)
    ch.get_oversample_experiment_name()

    print(settings["experiment"])

    dl = DataLoader(settings)
    #sequences, labels, demographics = dl.load_data()
    sequences, labels, demographics, full_data = dl.load_data() # need to modify the function to return also "full_data"

    # save sequences and labels and demographics
    # Save the variables

    with open('full_data.pkl', 'wb') as f:
        pickle.dump(full_data, f)

    with open('sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)

    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    with open('demographics.pkl', 'wb') as f:
        pickle.dump(demographics, f)


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

