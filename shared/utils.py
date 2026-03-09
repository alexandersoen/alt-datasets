"""Shared utility functions."""

from collections import Counter
from typing import Any, Generator, Iterator

Example = tuple[int, Any]
ExampleGenerator = Generator[Example, Any, None]
ExampleIterator = Iterator[Example]


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
