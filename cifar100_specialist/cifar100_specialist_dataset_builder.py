"""cifar100_specialist dataset."""

from itertools import tee
from typing import AbstractSet, Any, cast

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.image_classification.cifar import Cifar100

from shared.utils import ExampleGenerator, ignore_first_n, select_first_n

SEED = 42
NONSPECIALIST_PERC_LIST = [100, 20, 10]

MAMMAL_SPECIALIST_FINE_LABELS_BY_COARSE = {
  "aquatic_mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
  "large_carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
  "large_omnivores_and_herbivores": (
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
  ),
  "medium_mammals": ("fox", "porcupine", "possum", "raccoon", "skunk"),
  "small_mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
}


def _specialist_group_description() -> str:
  return ", ".join(MAMMAL_SPECIALIST_FINE_LABELS_BY_COARSE)


def _apply_specialist_filter(
  examples: ExampleGenerator,
  nonspecialist_perc: int,
  seed: int,
) -> ExampleGenerator:
  """Keep all specialist examples and a fraction of non-specialist examples."""
  rng = np.random.RandomState(seed=seed)
  nonspecialist_prob = nonspecialist_perc / 100.0

  for key, example in examples:
    if example["is_specialist"] or rng.binomial(1, nonspecialist_prob):
      yield key, example


def _annotate_specialist_flag(
  examples: ExampleGenerator,
  specialist_coarse_labels: AbstractSet[int],
) -> ExampleGenerator:
  """Annotate examples with specialist membership without filtering."""
  for key, example in examples:
    example["is_specialist"] = int(example["coarse_label"] in specialist_coarse_labels)
    yield key, example


class Cifar100SpecialistConfig(tfds.core.BuilderConfig):
  def __init__(self, *, nonspecialist_perc: int, **kwargs: Any) -> None:
    super().__init__(**kwargs)

    if nonspecialist_perc not in NONSPECIALIST_PERC_LIST:
      raise ValueError(f"Unsupported non-specialist percentage: {nonspecialist_perc}")

    self.nonspecialist_perc = nonspecialist_perc


def _make_builder_configs() -> list[Cifar100SpecialistConfig]:
  configs = []
  group_description = _specialist_group_description()
  for nonspecialist_perc in NONSPECIALIST_PERC_LIST:
    configs.append(
      Cifar100SpecialistConfig(
        name=f"nonspecialist_{nonspecialist_perc}",
        nonspecialist_perc=nonspecialist_perc,
        description=(
          "Mammal specialist over 25 CIFAR-100 classes "
          f"({group_description}) with {nonspecialist_perc}% of "
          "non-specialist examples retained."
        ),
      )
    )
  return configs


class Builder(Cifar100):
  """DatasetBuilder for cifar100_specialist dataset."""

  VERSION = tfds.core.Version("0.0.1")
  RELEASE_NOTES = {
    "0.0.1": "Initial release.",
  }
  BUILDER_CONFIGS = _make_builder_configs()

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
          "is_specialist": tfds.features.ClassLabel(num_classes=2),
        }
      ),
      supervised_keys=info.supervised_keys,
      homepage=info.homepage,
      citation=info.citation,
    )

  def _split_generators(
    self, dl_manager: download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """Returns SplitGenerators."""
    splits = super()._split_generators(dl_manager)
    build_config = cast(Cifar100SpecialistConfig, self.builder_config)

    features = cast(tfds.features.FeaturesDict, self.info.features)
    coarse_feature = cast(tfds.features.ClassLabel, features["coarse_label"])
    specialist_coarse_labels = {
      idx
      for idx, coarse_name in enumerate(coarse_feature.names)
      if coarse_name in MAMMAL_SPECIALIST_FINE_LABELS_BY_COARSE
    }

    train_gen, val_gen = tee(splits["train"], 2)
    res = {
      "train": ignore_first_n(train_gen, 50),
      "validation": select_first_n(val_gen, 50),
      "test": splits["test"],
    }

    for split_name, split_examples in res.items():
      res[split_name] = _annotate_specialist_flag(
        split_examples, specialist_coarse_labels
      )

    res["train"] = _apply_specialist_filter(
      res["train"],
      build_config.nonspecialist_perc,
      SEED,
    )

    return res
