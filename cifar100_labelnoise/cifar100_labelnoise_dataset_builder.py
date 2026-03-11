"""cifar100_labelnoise dataset.

TODO: Coarse labels are not noise.
"""

from itertools import tee
from typing import Any, cast

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import Cifar100

from shared.utils import ExampleGenerator, ignore_first_n, select_first_n

NUM_CLASSES = 100


def _apply_label_noise(
  examples: ExampleGenerator, num_noisy_classes: int, seed: int
) -> ExampleGenerator:
  """Apply paper-style label noise to classes 0..num_noisy_classes-1."""
  rng = np.random.RandomState(seed=seed)
  noise_classes = set(range(num_noisy_classes))

  for key, example in examples:
    if example["label"] in noise_classes:
      example["label"] = int(rng.randint(low=0, high=NUM_CLASSES))

    yield key, example


class Cifar100LabelNoiseConfig(tfds.core.BuilderConfig):
  def __init__(self, *, num_noisy_classes: int, **kwargs: Any) -> None:
    """BuilderConfig for cifar100_labelnoise.

    Args:
      num_noisy_classes: number of classes with label noise.
      **kwargs: keyword arguments forwarded to super.
    """
    super().__init__(**kwargs)
    self.num_noisy_classes = num_noisy_classes


class Builder(Cifar100):
  """DatasetBuilder for cifar100_labelnoise dataset."""

  VERSION = tfds.core.Version("0.0.4")
  BUILDER_CONFIGS = [
    Cifar100LabelNoiseConfig(
      name="first_0",
      num_noisy_classes=0,
      description="No noise.",
    ),
    Cifar100LabelNoiseConfig(
      name="first_10",
      num_noisy_classes=10,
      description="Label noise on classes 0-9.",
    ),
    Cifar100LabelNoiseConfig(
      name="first_25",
      num_noisy_classes=25,
      description="Label noise on classes 0-24.",
    ),
  ]
  SEED = 42

  def _split_generators(
    self, dl_manager: download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """Override to create train, validation, and test splits."""
    splits = super()._split_generators(dl_manager)
    build_config = cast(Cifar100LabelNoiseConfig, self.builder_config)

    train_gen, val_gen = tee(splits["train"], 2)
    res = {
      "train": ignore_first_n(train_gen, 50),
      "validation": select_first_n(val_gen, 50),
      "test": splits["test"],
    }

    res["train"] = _apply_label_noise(
      res["train"], build_config.num_noisy_classes, self.SEED
    )
    return res
