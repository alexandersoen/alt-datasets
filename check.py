"""
Script to check the statistics of the generated datasets.
"""

import tensorflow_datasets as tfds


def generate_hist(builder_str: str, split_str: str = "train") -> list[int]:
  builder = tfds.builder(builder_str)
  dataset = builder.as_dataset(split=split_str)

  hist = [0] * 100
  for example in dataset.as_numpy_iterator():  # pyright: ignore
    hist[int(example["label"])] += 1

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
