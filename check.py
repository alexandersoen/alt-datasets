"""
Script to check the statistics of the generated datasets.
"""

import tensorflow_datasets as tfds
from tensorflow_datasets.core import FileFormat


def generate_hist(builder_str: str, split_str: str = "train") -> list[int]:
  builder = tfds.builder(builder_str)
  if builder.info.file_format == FileFormat.ARRAY_RECORD:
    data = builder.as_data_source(split=split_str)
  else:
    data = builder.as_dataset(split=split_str).as_numpy_iterator()  # pyright: ignore

  hist = [0] * 100
  for example in data:
    hist[int(example["label"])] += 1  # pyright: ignore

  return hist


if __name__ == "__main__":
  print("Label Noise")
  print(generate_hist("cifar100_noisy/label_noise_0"))
  print(generate_hist("cifar100_noisy/label_noise_10"))
  print(generate_hist("cifar100_noisy/label_noise_25"))
  print()
  print("Longtail")
  print(generate_hist("cifar100_longtail/head_100"))
  print(generate_hist("cifar100_longtail/head_50"))
  print(generate_hist("cifar100_longtail/head_25"))
