"""Shared utility functions."""

from collections import Counter
from itertools import tee
from typing import Any, Generator, Iterator, Mapping, TypeAlias

ExampleRecord: TypeAlias = dict[str, Any]
ExampleKey: TypeAlias = Any
Example: TypeAlias = tuple[ExampleKey, ExampleRecord]
ExampleGenerator: TypeAlias = Generator[Example, Any, None]
ExampleIterator: TypeAlias = Iterator[Example]


def select_first_n(examples: ExampleIterator, n: int) -> ExampleGenerator:
  """Select the first n of each label class in a generator."""

  counter = Counter()
  for key, example in examples:
    label = example["label"]

    if counter[label] >= n:
      continue

    counter[label] += 1

    yield key, example


def ignore_first_n(examples: ExampleIterator, n: int) -> ExampleGenerator:
  """Ignore the first n of each label class in a generator."""

  counter = Counter()
  for key, example in examples:
    label = example["label"]

    if counter[label] < n:
      counter[label] += 1
      continue

    yield key, example


def split_train_validation(
  splits: Mapping[str, ExampleGenerator],
  validation_examples_per_class: int,
) -> dict[str, ExampleGenerator]:
  """Split a training generator into train/validation by label class."""
  train_gen, val_gen = tee(splits["train"], 2)
  return {
    "train": ignore_first_n(train_gen, validation_examples_per_class),
    "validation": select_first_n(val_gen, validation_examples_per_class),
    "test": splits["test"],
  }
