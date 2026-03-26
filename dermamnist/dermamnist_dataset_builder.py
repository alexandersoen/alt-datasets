"""DermaMNIST dataset."""

from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

from shared.utils import ExampleGenerator

_DERMAMNIST_URL = "https://zenodo.org/records/10519652/files/dermamnist.npz"


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for DermaMNIST dataset."""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
    "1.0.0": "Initial release.",
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict(
        {
          "image": tfds.features.Image(shape=(28, 28, 3)),
          "label": tfds.features.ClassLabel(num_classes=7),
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

    return {
      "train": self._generate_examples(
        raw_data["train_images"], raw_data["train_labels"]
      ),
      "validation": self._generate_examples(
        raw_data["val_images"], raw_data["val_labels"]
      ),
      "test": self._generate_examples(raw_data["test_images"], raw_data["test_labels"]),
    }

  def _generate_examples(
    self,
    images: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.integer[Any]],
  ) -> ExampleGenerator:
    """Yields examples."""
    for idx, (image, label) in enumerate(zip(images, labels)):
      yield (
        idx,
        {
          "image": image,
          "label": int(np.squeeze(label)),
        },
      )
