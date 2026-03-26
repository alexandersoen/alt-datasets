"""Calculate per-channel mean and std for a TFDS image dataset split."""

import argparse

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import FileFormat

from shared.utils import ExampleIterator


def create_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Print per-channel mean/std for a TFDS dataset split."
  )
  parser.add_argument("builder", help="TFDS builder string, e.g. pathmnist")
  parser.add_argument(
    "--split",
    default="train",
    help="Dataset split to use. Defaults to train.",
  )
  return parser


def create_data_iter(builder_str: str, split_str: str) -> ExampleIterator:
  builder = tfds.builder(builder_str)
  if builder.info.file_format == FileFormat.ARRAY_RECORD:
    return builder.as_data_source(split=split_str)

  return builder.as_dataset(split=split_str).as_numpy_iterator()  # pyright: ignore


def calculate_mean_std(builder_str: str, split_str: str) -> tuple[list[float], list[float]]:
  data_iter = create_data_iter(builder_str, split_str)

  channel_sum = None
  channel_sq_sum = None
  total_pixels = 0
  num_examples = 0

  for example in data_iter:
    image = np.asarray(example["image"], dtype=np.float64) / 255.0  # pyright: ignore
    if image.ndim == 2:
      image = image[..., None]

    if channel_sum is None:
      num_channels = image.shape[-1]
      channel_sum = np.zeros(num_channels, dtype=np.float64)
      channel_sq_sum = np.zeros(num_channels, dtype=np.float64)

    channel_sum += image.sum(axis=(0, 1))
    channel_sq_sum += np.square(image).sum(axis=(0, 1))
    total_pixels += image.shape[0] * image.shape[1]
    num_examples += 1

  if channel_sum is None or channel_sq_sum is None:
    raise ValueError(f"No examples found for {builder_str}:{split_str}")

  mean = channel_sum / total_pixels
  var = channel_sq_sum / total_pixels - np.square(mean)
  std = np.sqrt(np.maximum(var, 0.0))

  print(f"builder={builder_str}")
  print(f"split={split_str}")
  print(f"num_examples={num_examples}")
  print(f"mean={[round(x, 8) for x in mean.tolist()]}")
  print(f"std={[round(x, 8) for x in std.tolist()]}")

  return mean.tolist(), std.tolist()


if __name__ == "__main__":
  args = create_parser().parse_args()
  calculate_mean_std(args.builder, args.split)
