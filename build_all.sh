#!/bin/bash

# tfds build cifar100_labelnoise/
# tfds build cifar100_longtail/
# tfds build cifar100_specialist/

tfds build --overwrite --file_format=array_record cifar100_labelnoise/
tfds build --overwrite --file_format=array_record cifar100_longtail/
tfds build --overwrite --file_format=array_record cifar100_specialist/

# tfds build --file_format=array_record --datasets=imagenet2012 --manual_dir=/data/giil/tensorflow_datasets/downloads/manual
