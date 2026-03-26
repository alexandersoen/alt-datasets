"""cifar100_longtail dataset."""

from typing import Any, cast

import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import Cifar100

from shared.corrupt import annotate_binary_flag, cap_examples_per_class
from shared.utils import ExampleGenerator, split_train_validation

NUM_CLASSES = 100
EXAMPLES_PER_CLASS = 500


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
    splits = cast(dict[str, ExampleGenerator], super()._split_generators(dl_manager))
    build_config = cast(Cifar100LongtailConfig, self.builder_config)

    def target_count_fn(label: int) -> int:
      return (
        build_config.head_size
        if label < build_config.num_head_classes
        else build_config.tail_size
      )

    res = split_train_validation(splits, validation_examples_per_class=50)

    head_classes = set(range(build_config.num_head_classes))
    for split_name, split_examples in res.items():
      res[split_name] = annotate_binary_flag(
        split_examples,
        "is_head",
        lambda example: example["label"] in head_classes,
      )
      res[split_name] = cap_examples_per_class(
        res[split_name],
        label_field="label",
        target_count_fn=target_count_fn,
      )

    return res
