"""cifar100_specialist dataset."""

from . import cifar100_specialist_dataset_builder
import tensorflow_datasets as tfds

class Cifar100SpecialistTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for cifar100_specialist dataset."""

  DATASET_CLASS = cifar100_specialist_dataset_builder.Builder
  SPLITS = {
      "train": 3,
      "test": 1,
  }

if __name__ == "__main__":
  tfds.testing.test_main()
