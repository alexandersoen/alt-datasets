"""Shared filtering and corruption helpers for dataset builders."""

from collections import Counter
from typing import Callable

import numpy as np

from shared.utils import ExampleGenerator, ExampleRecord


def annotate_binary_flag(
  examples: ExampleGenerator,
  field_name: str,
  predicate: Callable[[ExampleRecord], bool],
) -> ExampleGenerator:
  """Annotate examples with a binary flag computed from each example."""
  for key, example in examples:
    example[field_name] = int(predicate(example))
    yield key, example


def annotate_constant_flag(
  examples: ExampleGenerator,
  field_name: str,
  value: int = 0,
) -> ExampleGenerator:
  """Annotate examples with a constant binary flag."""
  for key, example in examples:
    example[field_name] = value
    yield key, example


def keep_specialist_and_sample_rest(
  examples: ExampleGenerator,
  *,
  specialist_field: str,
  nonspecialist_perc: int,
  seed: int,
) -> ExampleGenerator:
  """Keep all specialist examples and sample the remaining examples."""
  rng = np.random.RandomState(seed=seed)
  nonspecialist_prob = nonspecialist_perc / 100.0

  for key, example in examples:
    if example[specialist_field] or rng.binomial(1, nonspecialist_prob):
      yield key, example


def cap_examples_per_class(
  examples: ExampleGenerator,
  *,
  label_field: str,
  target_count_fn: Callable[[int], int],
) -> ExampleGenerator:
  """Keep at most the configured number of examples for each class."""
  counter = Counter()
  for key, example in examples:
    label = int(example[label_field])
    target_count = target_count_fn(label)

    if counter[label] >= target_count:
      continue

    counter[label] += 1
    yield key, example


def apply_uniform_label_noise(
  examples: ExampleGenerator,
  *,
  label_field: str,
  noise_flag_field: str,
  noisy_labels: set[int],
  num_classes: int,
  seed: int,
) -> ExampleGenerator:
  """Randomly relabel selected classes and mark which examples changed."""
  rng = np.random.RandomState(seed=seed)

  for key, example in examples:
    original_label = int(example[label_field])
    is_corrupted = 0

    if original_label in noisy_labels:
      new_label = int(rng.randint(low=0, high=num_classes))
      is_corrupted = int(new_label != original_label)
      example[label_field] = new_label

    example[noise_flag_field] = is_corrupted
    yield key, example
