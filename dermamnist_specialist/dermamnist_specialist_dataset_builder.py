"""DermaMNIST specialist dataset."""

from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

from shared.utils import ExampleGenerator

_DERMAMNIST_URL = "https://zenodo.org/records/10519652/files/dermamnist.npz"
_SEED = 42
_NONSPECIALIST_PERC_LIST = [100, 20, 10]

# Official DermaMNIST labels from MedMNIST v2:
# 0 actinic keratoses and intraepithelial carcinoma,
# 1 basal cell carcinoma, 2 benign keratosis-like lesions,
# 3 dermatofibroma, 4 melanoma, 5 melanocytic nevi, 6 vascular lesions.
_DERMAMNIST_LABELS = (
  "actinic keratoses and intraepithelial carcinoma",
  "basal cell carcinoma",
  "benign keratosis-like lesions",
  "dermatofibroma",
  "melanoma",
  "melanocytic nevi",
  "vascular lesions",
)
_DERMAMNIST_SPECIALIST_LABELS = frozenset((4, 5))


class DermaMNISTSpecialistConfig(tfds.core.BuilderConfig):
  def __init__(self, *, nonspecialist_perc: int, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    if nonspecialist_perc not in _NONSPECIALIST_PERC_LIST:
      raise ValueError(f"Unsupported non-specialist percentage: {nonspecialist_perc}")
    self.nonspecialist_perc = nonspecialist_perc


def _make_builder_configs() -> list[DermaMNISTSpecialistConfig]:
  return [
    DermaMNISTSpecialistConfig(
      name=f"nonspecialist_{nonspecialist_perc}",
      nonspecialist_perc=nonspecialist_perc,
      description=(
        "DermaMNIST melanocytic-lesion specialist over melanoma and "
        f"melanocytic nevi with {nonspecialist_perc}% of non-specialist "
        "training examples retained."
      ),
    )
    for nonspecialist_perc in _NONSPECIALIST_PERC_LIST
  ]


def _annotate_specialist_flag(
  images: npt.NDArray[np.uint8],
  labels: npt.NDArray[np.integer[Any]],
) -> ExampleGenerator:
  for idx, (image, label) in enumerate(zip(images, labels)):
    class_id = int(np.squeeze(label))
    yield (
      idx,
      {
        "image": image,
        "label": class_id,
        "is_specialist": int(class_id in _DERMAMNIST_SPECIALIST_LABELS),
      },
    )


def _filter_specialist_examples(
  examples: ExampleGenerator,
  nonspecialist_perc: int,
  seed: int,
) -> ExampleGenerator:
  rng = np.random.RandomState(seed=seed)
  nonspecialist_prob = nonspecialist_perc / 100.0

  for key, example in examples:
    if example["is_specialist"] or rng.binomial(1, nonspecialist_prob):
      yield key, example


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dermamnist_specialist dataset."""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
    "1.0.0": "Initial release.",
  }
  BUILDER_CONFIGS = _make_builder_configs()

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict(
        {
          "image": tfds.features.Image(shape=(28, 28, 3)),
          "label": tfds.features.ClassLabel(names=_DERMAMNIST_LABELS),
          "is_specialist": tfds.features.ClassLabel(names=("no", "yes")),
        }
      ),
      supervised_keys=("image", "label"),
      homepage="https://medmnist.com/",
    )

  def _split_generators(
    self, dl_manager: tfds.download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """Returns SplitGenerators."""
    npz_path = dl_manager.download(_DERMAMNIST_URL)

    with tf.io.gfile.GFile(npz_path, "rb") as f:
      raw_data = np.load(cast(BinaryIO, f))

    build_config = cast(DermaMNISTSpecialistConfig, self.builder_config)

    return {
      "train": self._generate_examples(
        raw_data["train_images"],
        raw_data["train_labels"],
        nonspecialist_perc=build_config.nonspecialist_perc,
      ),
      "validation": self._generate_examples(
        raw_data["val_images"],
        raw_data["val_labels"],
      ),
      "test": self._generate_examples(
        raw_data["test_images"],
        raw_data["test_labels"],
      ),
    }

  def _generate_examples(
    self,
    images: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.integer[Any]],
    nonspecialist_perc: int | None = None,
  ) -> ExampleGenerator:
    """Yields examples."""
    examples = _annotate_specialist_flag(images, labels)
    if nonspecialist_perc is not None:
      examples = _filter_specialist_examples(examples, nonspecialist_perc, _SEED)

    yield from examples
