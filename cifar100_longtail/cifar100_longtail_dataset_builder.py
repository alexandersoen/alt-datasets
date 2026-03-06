"""cifar100_longtail dataset."""

from collections import Counter
from typing import Any, Generator, cast

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import (
  _CIFAR_IMAGE_SHAPE,
  Cifar100,
)

SEED = 42
NUM_CLASSES = 100

ExampleGenerator = Generator[tuple[int, Any], Any, None]


class Cifar100LongtailConfig(tfds.core.BuilderConfig):
  def __init__(
    self,
    *,
    num_head_classes: int,
    head_size: int = 500,
    tail_size: int = 50,
    **kwargs: Any,
  ) -> None:
    """BuilderConfig for cifar100_longtail.

    Args:
      num_head_classes: number of labels that are head class.
      head_size: num of examples in a head class.
      tail_size: num of examples in a tail class.
      **kwargs: keyword arguments forwarded to super.
    """
    super().__init__(**kwargs)
    self.num_head_classes = num_head_classes
    self.head_size = head_size
    self.tail_size = tail_size


class Builder(Cifar100):
  """DatasetBuilder for cifar100_label_noise dataset."""

  VERSION = tfds.core.Version("0.0.2")
  BUILDER_CONFIGS = [
    Cifar100LongtailConfig(name="head_100", num_head_classes=100),
    Cifar100LongtailConfig(name="head_50", num_head_classes=50),
    Cifar100LongtailConfig(name="head_25", num_head_classes=25),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      description="The CIFAR-100 filtered to long-tail dataset.",
      features=tfds.features.FeaturesDict(
        {
          "id": tfds.features.Text(),
          "image": tfds.features.Image(shape=_CIFAR_IMAGE_SHAPE),
          "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
          "coarse_label": tfds.features.ClassLabel(num_classes=20),
        }
      ),
      supervised_keys=("image", "label"),
    )

  def _split_generators(
    self, dl_manager: download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """ """
    return super()._split_generators(dl_manager)

  def _generate_examples(
    self, split_prefix: str, filepaths: list[str]
  ) -> ExampleGenerator:
    """ """
    rng = np.random.RandomState(seed=SEED)
    build_config = cast(Cifar100LongtailConfig, self.builder_config)

    # Just read into memory as cifar100 is "small"
    all_examples = list(super()._generate_examples(split_prefix, filepaths))

    class_order = np.arange(NUM_CLASSES)
    rng.shuffle(class_order)
    head_classes = set(class_order[: build_config.num_head_classes])

    counter = Counter()
    for key, example in all_examples:
      label = example["label"]

      if label in head_classes:
        target_count = build_config.head_size
      else:
        target_count = build_config.tail_size

      if counter[label] >= target_count:
        continue

      counter[label] += 1

      yield key, example
