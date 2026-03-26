"""Shared helpers for MedMNIST-style dataset builders."""

from typing import Any, BinaryIO, Callable, Mapping, cast

import numpy as np
import numpy.typing as npt
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

from shared.utils import ExampleGenerator

ExtraFieldsFn = Callable[[int], Mapping[str, Any]]


def load_npz(
  dl_manager: tfds.download.DownloadManager,
  url: str,
) -> Any:
  """Download and load an NPZ archive."""
  npz_path = dl_manager.download(url)
  with tf.io.gfile.GFile(npz_path, "rb") as f:
    return np.load(cast(BinaryIO, f))


def generate_examples(
  images: npt.NDArray[np.uint8],
  labels: npt.NDArray[np.integer[Any]],
  *,
  extra_fields_fn: ExtraFieldsFn | None = None,
) -> ExampleGenerator:
  """Yield MedMNIST examples from paired image and label arrays."""
  for idx, (image, label) in enumerate(zip(images, labels)):
    class_id = int(np.squeeze(label))
    example = {
      "image": image,
      "label": class_id,
    }
    if extra_fields_fn is not None:
      example.update(extra_fields_fn(class_id))
    yield idx, example


def build_splits(
  raw_data: Any,
  generate_examples_fn: Callable[..., ExampleGenerator],
  *,
  train_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, ExampleGenerator]:
  """Build standard train/validation/test splits from MedMNIST arrays."""
  train_kwargs = dict(train_kwargs or {})
  return {
    "train": generate_examples_fn(
      raw_data["train_images"],
      raw_data["train_labels"],
      **train_kwargs,
    ),
    "validation": generate_examples_fn(
      raw_data["val_images"],
      raw_data["val_labels"],
    ),
    "test": generate_examples_fn(
      raw_data["test_images"],
      raw_data["test_labels"],
    ),
  }
