"""
Script to check the statistics of the generated datasets.
"""

import tensorflow_datasets as tfds
from tensorflow_datasets.core import FileFormat


def generate_hist(builder_str: str, split_str: str = "train") -> list[int]:
  print(builder_str)
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
  print("============= TRAIN =============")
  print("Label Noise")
  print(generate_hist("cifar100_labelnoise/first_0"))
  print(generate_hist("cifar100_labelnoise/first_10"))
  print(generate_hist("cifar100_labelnoise/first_25"))
  print()
  print("Longtail")
  print(generate_hist("cifar100_longtail/head_100"))
  print(generate_hist("cifar100_longtail/head_50"))
  print(generate_hist("cifar100_longtail/head_25"))
  print()
  print("Specialist")
  print(generate_hist("cifar100_specialist/nonspecialist_100"))
  print(generate_hist("cifar100_specialist/nonspecialist_20"))
  print(generate_hist("cifar100_specialist/nonspecialist_10"))
  print("============= TRAIN =============\n")

  print("============= VAL =============")
  print("Label Noise")
  print(generate_hist("cifar100_labelnoise/first_0", "validation"))
  print(generate_hist("cifar100_labelnoise/first_10", "validation"))
  print(generate_hist("cifar100_labelnoise/first_25", "validation"))
  print()
  print("Longtail")
  print(generate_hist("cifar100_longtail/head_100", "validation"))
  print(generate_hist("cifar100_longtail/head_50", "validation"))
  print(generate_hist("cifar100_longtail/head_25", "validation"))
  print()
  print("Specialist")
  print(generate_hist("cifar100_specialist/nonspecialist_100", "validation"))
  print(generate_hist("cifar100_specialist/nonspecialist_20", "validation"))
  print(generate_hist("cifar100_specialist/nonspecialist_10", "validation"))
  print("============= VAL =============\n")

  print("============= TEST =============")
  print("Label Noise")
  print(generate_hist("cifar100_labelnoise/first_0", "test"))
  print(generate_hist("cifar100_labelnoise/first_10", "test"))
  print(generate_hist("cifar100_labelnoise/first_25", "test"))
  print()
  print("Longtail")
  print(generate_hist("cifar100_longtail/head_100", "test"))
  print(generate_hist("cifar100_longtail/head_50", "test"))
  print(generate_hist("cifar100_longtail/head_25", "test"))
  print()
  print("Specialist")
  print(generate_hist("cifar100_specialist/nonspecialist_100", "test"))
  print(generate_hist("cifar100_specialist/nonspecialist_20", "test"))
  print(generate_hist("cifar100_specialist/nonspecialist_10", "test"))
  print("============= TEST =============\n")
