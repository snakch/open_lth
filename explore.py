import models.registry
import cli.runner_registry
import platforms.registry
from training.desc import TrainingDesc
import numpy as np


default_hparams = models.registry.get_default_hparams("cifar_vgg_16")

default_hparams.model_hparams

model = models.registry.get(default_hparams.model_hparams, outputs=10)

total = 0
for _, v in model.named_parameters():
    total += np.product(list(v.shape))
total


model = models.registry.load(
    "/home/simon_wsl/data/open_lth_data/train_7312e802e619673d23c7a02eba8aee52/replicate_1/main/checkpoint.pth",
    ,
    model_hparams,
    10,
)


model.layers

import argparse

parser = argparse.ArgumentParser()
rname = "train"
platforms.registry.get("local").add_args(parser)
cli.runner_registry.get(rname).add_args(parser)

args = parser.parse_args()


runner = cli.runner_registry.get("train")

runner.create_from_args(args)

models.registry.get()


hparams = models.registry.get_default_hparams("cifar_vgg_16")

models.registry.get(hparams)


hparams = models.registry.registered_models[2].default_hparams()


hparams

is_valid_model_name("cifar_vgg_16")


hparams.model_name


fr
