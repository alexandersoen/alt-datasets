"""DermaMNIST specialist dataset."""

from typing import Any, cast

import tensorflow_datasets.public_api as tfds

from shared.corrupt import keep_specialist_and_sample_rest
from shared.medmnist import build_splits, generate_examples, load_npz
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
    raw_data = load_npz(dl_manager, _DERMAMNIST_URL)
    build_config = cast(DermaMNISTSpecialistConfig, self.builder_config)
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
        "is_specialist": int(class_id in _DERMAMNIST_SPECIALIST_LABELS)
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
