"""cifar100_labelnoise dataset.

TODO: Coarse labels are not noise.
"""

from typing import Any, cast

import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import Cifar100

from shared.corrupt import (
  annotate_constant_flag,
  apply_uniform_label_noise,
)
from shared.utils import ExampleGenerator, split_train_validation

NUM_CLASSES = 100


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

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    info = super()._info()
    inherited_features = cast(tfds.features.FeaturesDict, info.features)
    return tfds.core.DatasetInfo(
      builder=self,
      description=info.description,
      features=tfds.features.FeaturesDict(
        {
          **inherited_features,
          "is_noise": tfds.features.ClassLabel(num_classes=2),
        }
      ),
      supervised_keys=info.supervised_keys,
      homepage=info.homepage,
      citation=info.citation,
    )

  def _split_generators(
    self, dl_manager: download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """Override to create train, validation, and test splits."""
    splits = cast(dict[str, ExampleGenerator], super()._split_generators(dl_manager))
    build_config = cast(Cifar100LabelNoiseConfig, self.builder_config)

    res = split_train_validation(splits, validation_examples_per_class=50)

    for split_name, split_examples in res.items():
      res[split_name] = annotate_constant_flag(split_examples, "is_noise")
      res[split_name] = apply_uniform_label_noise(
        res[split_name],
        label_field="label",
        noise_flag_field="is_noise",
        noisy_labels=set(range(build_config.num_noisy_classes)),
        num_classes=NUM_CLASSES,
        seed=self.SEED,
      )

    return res
