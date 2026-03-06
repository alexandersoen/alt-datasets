"""imagenet2012_dogs dataset."""

import io
import random
from typing import cast

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.datasets.imagenet2012 import imagenet_common
from tensorflow_datasets.datasets.imagenet2012.imagenet2012_dataset_builder import (
    Builder as Imagenet2012Builder,
)

_VERSION = tfds.core.Version("0.0.1")
_RELEASE_NOTES = {
    "0.0.1": "Initial dataset",
}

N_CLASSES = 1_000
NONDOG_PERC_LIST = [100, 4, 2]
MAX_EXAMPLES_PER_CLASS = 1_350

DOGS_SET = set(
    [
        "n02110627",
        "n02088094",
        "n02116738",
        "n02096051",
        "n02093428",
        "n02107908",
        "n02096294",
        "n02110806",
        "n02088238",
        "n02088364",
        "n02093647",
        "n02107683",
        "n02089078",
        "n02086646",
        "n02088466",
        "n02088632",
        "n02106166",
        "n02093754",
        "n02090622",
        "n02096585",
        "n02106382",
        "n02108089",
        "n02112706",
        "n02105251",
        "n02101388",
        "n02108422",
        "n02096177",
        "n02113186",
        "n02099849",
        "n02085620",
        "n02112137",
        "n02101556",
        "n02102318",
        "n02106030",
        "n02099429",
        "n02096437",
        "n02115913",
        "n02115641",
        "n02107142",
        "n02089973",
        "n02100735",
        "n02102040",
        "n02108000",
        "n02109961",
        "n02099267",
        "n02108915",
        "n02106662",
        "n02100236",
        "n02097130",
        "n02099601",
        "n02101006",
        "n02109047",
        "n02111500",
        "n02107574",
        "n02105056",
        "n02091244",
        "n02100877",
        "n02093991",
        "n02102973",
        "n02090721",
        "n02091032",
        "n02085782",
        "n02112350",
        "n02105412",
        "n02093859",
        "n02105505",
        "n02104029",
        "n02099712",
        "n02095570",
        "n02111129",
        "n02098413",
        "n02110063",
        "n02105162",
        "n02085936",
        "n02113978",
        "n02107312",
        "n02113712",
        "n02097047",
        "n02111277",
        "n02094114",
        "n02091467",
        "n02094258",
        "n02105641",
        "n02091635",
        "n02086910",
        "n02086079",
        "n02113023",
        "n02112018",
        "n02110958",
        "n02090379",
        "n02087394",
        "n02106550",
        "n02109525",
        "n02091831",
        "n02111889",
        "n02104365",
        "n02097298",
        "n02092002",
        "n02095889",
        "n02105855",
        "n02086240",
        "n02110185",
        "n02097658",
        "n02098105",
        "n02093256",
        "n02113799",
        "n02097209",
        "n02102480",
        "n02108551",
        "n02097474",
        "n02113624",
        "n02087046",
        "n02100583",
        "n02089867",
        "n02092339",
        "n02102177",
        "n02098286",
        "n02091134",
        "n02095314",
        "n02094433",
    ]
)


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
    def __init__(self, nondog_perc: int, seed: int) -> None:
        self.nondog_perc = nondog_perc / 100.0

        self.seed = seed

        random.seed(seed)

    def is_dog(self, record) -> bool:
        return record["label"] in DOGS_SET

    def label_is_dog(self, record):
        record["is_dog"] = int(self.is_dog(record))
        return record

    def filter(self, record) -> bool:
        is_not_dog = not bool(record["is_dog"])
        return is_not_dog and bool(random.random() > self.nondog_perc)


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
        build_config = cast(Imagenet2012DogsConfig, self.builder_config)

        if validation_labels is None and labels_exist:
            gen_fn = self._filtered_generate_examples(
                archive, perc=build_config.nondog_perc / 100.0, seed=self.SEED
            )

            for key, example in gen_fn:
                yield key, example
        else:
            gen_fn = super()._generate_examples(
                archive,
                validation_labels=validation_labels,
                labels_exist=labels_exist,
            )

            nondog_downsampler = NonDogDownSampler(
                nondog_perc=build_config.nondog_perc,
                seed=self.SEED,
            )

            for key, example in gen_fn:
                example = nondog_downsampler.label_is_dog(example)

                if (
                    labels_exist
                    and build_config.nondog_perc != 100
                    and nondog_downsampler.filter(example)
                ):
                    continue

                yield key, example

    def _filtered_generate_examples(self, archive, perc: float, seed: int):
        np.random.seed(seed)

        for fname, fobj in archive:
            label = fname[:-4]
            is_dog = label in DOGS_SET
            fobj_mem = io.BytesIO(fobj.read())

            iter_archive = tfds.download.iter_archive(
                fobj_mem, tfds.download.ExtractMethod.TAR_STREAM
            )

            if is_dog:
                includes = np.ones(MAX_EXAMPLES_PER_CLASS)
            else:
                includes = np.random.binomial(
                    1, perc, size=MAX_EXAMPLES_PER_CLASS
                )

            for (image_fname, image), inc in zip(iter_archive, includes):
                if not inc:
                    continue

                image = super()._fix_image(image_fname, image)
                record = {
                    "file_name": image_fname,
                    "image": image,
                    "label": label,
                    "is_dog": int(is_dog),
                }
                yield image_fname, record
