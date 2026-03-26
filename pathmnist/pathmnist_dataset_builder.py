"""PathMNIST dataset."""

import tensorflow_datasets.public_api as tfds

from shared.medmnist import build_splits, generate_examples, load_npz
from shared.utils import ExampleGenerator

_PATHMNIST_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz"


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for PathMNIST dataset."""

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
          "label": tfds.features.ClassLabel(num_classes=9),
        }
      ),
      supervised_keys=("image", "label"),
      homepage="https://medmnist.com/",
    )

  def _split_generators(
    self, dl_manager: tfds.download.DownloadManager
  ) -> dict[str, ExampleGenerator]:
    """Returns SplitGenerators."""
    raw_data = load_npz(dl_manager, _PATHMNIST_URL)
    return build_splits(raw_data, generate_examples)
