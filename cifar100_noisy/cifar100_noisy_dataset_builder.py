"""cifar100_noisy dataset.

TODO: Coarse labels are not noise.
"""

import numpy as np
from typing import Generator, cast, Any

from tensorflow_datasets.core import download
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification.cifar import (
  _CIFAR_IMAGE_SHAPE,
  Cifar100,
)

SEED = 42
NUM_CLASSES = 100

ExampleGenerator = Generator[tuple[int, Any], Any, None]


class Cifar100NoisyConfig(tfds.core.BuilderConfig):
  def __init__(self, *, num_noisy_classes: int, **kwargs: Any) -> None:
    """BuilderConfig for cifar100_label_noise.

    Args:
      num_noisy_classes: number of classes with label noise.
      **kwargs: keyword arguments forwarded to super.
    """
    super().__init__(**kwargs)
    self.num_noisy_classes = num_noisy_classes


class Builder(Cifar100):
  """DatasetBuilder for cifar100_label_noise dataset."""

  VERSION = tfds.core.Version("0.0.2")
  BUILDER_CONFIGS = [
    Cifar100NoisyConfig(name="noise_0", num_noisy_classes=0, description="No noise."),
    Cifar100NoisyConfig(
      name="noise_10", num_noisy_classes=10, description="10 classes are noise."
    ),
    Cifar100NoisyConfig(
      name="noise_25", num_noisy_classes=25, description="25 classes are noise."
    ),
  ]
  SEED = 42

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      description="The CIFAR-100 with label noise.",
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
    build_config = cast(Cifar100NoisyConfig, self.builder_config)

    gen_fn = super()._generate_examples(split_prefix, filepaths)

    class_order = np.arange(NUM_CLASSES)
    rng.shuffle(class_order)
    noise_classes = set(class_order[: build_config.num_noisy_classes])

    for key, example in gen_fn:
      if example["label"] in noise_classes:
        example["label"] = rng.randint(low=0, high=NUM_CLASSES)

      yield (key, example)
