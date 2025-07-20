"""imagenet2012_dogs dataset."""

import random
from typing import cast

import tensorflow_datasets as tfds
from tensorflow_datasets.datasets.imagenet2012 import imagenet_common
from tensorflow_datasets.datasets.imagenet2012.imagenet2012_dataset_builder import (
    Builder as Imagenet2012Builder,
)

_VERSION = tfds.core.Version("0.0.1")
_RELEASE_NOTES = {
    "0.0.1": "Initial dataset",
}

N_CLASSES = 1000
NONDOG_PERC_LIST = [100, 4, 2]


class Imagenet2012DogsConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cifar100Longtail."""

    def __init__(self, *, nondog_perc=100, **kwargs):
        super().__init__(**kwargs)
        self.nondog_perc = nondog_perc


def _make_builder_configs():
    config_list = []
    for nondog_perc in NONDOG_PERC_LIST:
        name_str = f"nondog_{nondog_perc}"
        description_str = f"percentage of nondog sample = {nondog_perc}"
        config_list.append(
            Imagenet2012DogsConfig(
                name=name_str,
                version=_VERSION,
                release_notes=_RELEASE_NOTES,
                description=description_str,
                nondog_perc=nondog_perc,
            )
        )
    return config_list


class NonDogDownSampler:
    LOWER_DOG_SYNSET_ID = 2085620
    UPPER_DOG_SYNSET_ID = 2120079

    def __init__(self, nondog_perc: int, seed: int) -> None:
        self.nondog_perc = nondog_perc / 100.0

        self.seed = seed

        random.seed(seed)

    def is_dog(self, record) -> bool:
        file_name = record["file_name"].decode("utf-8")
        synset_id_str, *_ = file_name.split("_")
        synset_id = int(synset_id_str[1:])

        return (
            self.LOWER_DOG_SYNSET_ID <= synset_id <= self.UPPER_DOG_SYNSET_ID
        )

    def label_is_dog(self, record):
        record["is_dog"] = int(self.is_dog(record))
        return record

    def filter(self, record) -> bool:
        is_dog = record["is_dog"]
        if not is_dog and random.uniform(0, 1) < self.nondog_perc:
            return False

        return True


class Builder(Imagenet2012Builder):
    """DatasetBuilder for cifar100_label_noise dataset."""

    BUILDER_CONFIGS = _make_builder_configs()
    SEED = 42

    def _info(self):
        names_file = imagenet_common.label_names_file()
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(encoding_format="jpeg"),
                    "label": tfds.features.ClassLabel(names_file=names_file),
                    "file_name": tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
                    "is_dog": tfds.features.ClassLabel(num_classes=2),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://image-net.org/",
        )

    def _split_generators(self, dl_manager):
        return super()._split_generators(dl_manager)

    def _generate_examples(
        self, archive, validation_labels=None, labels_exist=True
    ):
        gen_fn = super()._generate_examples(
            archive,
            validation_labels=validation_labels,
            labels_exist=labels_exist,
        )

        build_config = cast(Imagenet2012DogsConfig, self.builder_config)
        nondog_downsampler = NonDogDownSampler(
            nondog_perc=build_config.nondog_perc,
            seed=self.SEED,
        )

        for key, example in gen_fn:
            example = nondog_downsampler.label_is_dog(example)

            if build_config.nondog_perc != 100 and nondog_downsampler.filter(
                example
            ):
                continue

            yield key, example
