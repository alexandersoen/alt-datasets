"""cifar100_longtail dataset."""

from collections import Counter
from itertools import tee
from typing import Any, cast

import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import Cifar100

from shared.utils import ExampleGenerator, ignore_first_n, select_first_n

NUM_CLASSES = 100
EXAMPLES_PER_CLASS = 500


def _apply_longtail_filter(
  examples: ExampleGenerator,
  num_head_classes: int,
  head_size: int,
  tail_size: int,
) -> ExampleGenerator:
  """Apply longtail filtering with head classes 0..num_head_classes-1."""
  head_classes = set(range(num_head_classes))

  counter = Counter()
  for key, example in examples:
    label = example["label"]

    if label in head_classes:
      target_count = head_size
    else:
      target_count = tail_size

    if counter[label] >= target_count:
      continue

    counter[label] += 1

    yield key, example


def _annotate_head_flag(
  examples: ExampleGenerator,
  head_classes: set[int],
) -> ExampleGenerator:
  """Annotate examples with head-class membership without filtering."""
  for key, example in examples:
    example["is_head"] = int(example["label"] in head_classes)
    yield key, example


class Cifar100LongtailConfig(tfds.core.BuilderConfig):
  def __init__(
    self,
    *,
    num_head_classes: int,
    head_size: int = 450,
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

    if head_size + tail_size != EXAMPLES_PER_CLASS:
      raise ValueError(
        "Head size + tail size does not equal number of examples per class"
      )

    self.num_head_classes = num_head_classes
    self.head_size = head_size
    self.tail_size = tail_size


class Builder(Cifar100):
  """DatasetBuilder for cifar100_longtail dataset."""

  VERSION = tfds.core.Version("0.0.4")
  BUILDER_CONFIGS = [
    Cifar100LongtailConfig(name="head_100", num_head_classes=100),
    Cifar100LongtailConfig(name="head_50", num_head_classes=50),
    Cifar100LongtailConfig(name="head_25", num_head_classes=25),
  ]

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
          "is_head": tfds.features.ClassLabel(num_classes=2),
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
    splits = super()._split_generators(dl_manager)
    build_config = cast(Cifar100LongtailConfig, self.builder_config)

    train_gen, val_gen = tee(splits["train"], 2)
    res = {
      "train": ignore_first_n(train_gen, 50),
      "validation": select_first_n(val_gen, 50),
      "test": splits["test"],
    }

    head_classes = set(range(build_config.num_head_classes))
    for split_name, split_examples in res.items():
      res[split_name] = _annotate_head_flag(split_examples, head_classes)

    res["train"] = _apply_longtail_filter(
      res["train"],
      build_config.num_head_classes,
      build_config.head_size,
      build_config.tail_size,
    )

    return res
