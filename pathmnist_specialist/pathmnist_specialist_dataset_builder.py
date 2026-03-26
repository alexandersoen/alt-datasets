"""PathMNIST specialist dataset."""

from typing import Any, cast

import tensorflow_datasets.public_api as tfds

from shared.corrupt import keep_specialist_and_sample_rest
from shared.medmnist import build_splits, generate_examples, load_npz
from shared.utils import ExampleGenerator

_PATHMNIST_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz"
_SEED = 42
_NONSPECIALIST_PERC_LIST = [100, 20, 10]

# Official PathMNIST labels from MedMNIST v2:
# 0 adipose, 1 background, 2 debris, 3 lymphocytes, 4 mucus,
# 5 smooth muscle, 6 normal colon mucosa, 7 cancer-associated stroma,
# 8 colorectal adenocarcinoma epithelium.
_PATHMNIST_LABELS = (
  "adipose",
  "background",
  "debris",
  "lymphocytes",
  "mucus",
  "smooth muscle",
  "normal colon mucosa",
  "cancer-associated stroma",
  "colorectal adenocarcinoma epithelium",
)
_PATHMNIST_SPECIALIST_LABELS = frozenset((3, 7, 8))


class PathMNISTSpecialistConfig(tfds.core.BuilderConfig):
  def __init__(self, *, nonspecialist_perc: int, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    if nonspecialist_perc not in _NONSPECIALIST_PERC_LIST:
      raise ValueError(f"Unsupported non-specialist percentage: {nonspecialist_perc}")
    self.nonspecialist_perc = nonspecialist_perc


def _make_builder_configs() -> list[PathMNISTSpecialistConfig]:
  return [
    PathMNISTSpecialistConfig(
      name=f"nonspecialist_{nonspecialist_perc}",
      nonspecialist_perc=nonspecialist_perc,
      description=(
        "PathMNIST colorectal tumor microenvironment specialist over "
        "lymphocytes, cancer-associated stroma, and colorectal "
        f"adenocarcinoma epithelium with {nonspecialist_perc}% of "
        "non-specialist training examples retained."
      ),
    )
    for nonspecialist_perc in _NONSPECIALIST_PERC_LIST
  ]
class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pathmnist_specialist dataset."""

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
          "label": tfds.features.ClassLabel(names=_PATHMNIST_LABELS),
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
    raw_data = load_npz(dl_manager, _PATHMNIST_URL)
    build_config = cast(PathMNISTSpecialistConfig, self.builder_config)
    return build_splits(
      raw_data,
      self._generate_examples,
      train_kwargs={"nonspecialist_perc": build_config.nonspecialist_perc},
    )

  def _generate_examples(
    self,
    images: Any,
    labels: Any,
    nonspecialist_perc: int | None = None,
  ) -> ExampleGenerator:
    """Yields examples."""
    examples = generate_examples(
      images,
      labels,
      extra_fields_fn=lambda class_id: {
        "is_specialist": int(class_id in _PATHMNIST_SPECIALIST_LABELS)
      },
    )
    if nonspecialist_perc is not None:
      examples = keep_specialist_and_sample_rest(
        examples,
        specialist_field="is_specialist",
        nonspecialist_perc=nonspecialist_perc,
        seed=_SEED,
      )

    yield from examples
