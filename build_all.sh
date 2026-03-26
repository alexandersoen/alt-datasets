#!/bin/bash

# tfds build cifar100_labelnoise/
# tfds build cifar100_longtail/
# tfds build cifar100_specialist/
# tfds build pathmnist/
# tfds build dermamnist/

tfds build --overwrite --file_format=array_record cifar100_labelnoise/
tfds build --overwrite --file_format=array_record cifar100_longtail/
tfds build --overwrite --file_format=array_record cifar100_specialist/
tfds build --overwrite --file_format=array_record pathmnist/
tfds build --overwrite --file_format=array_record dermamnist/
tfds build --overwrite --file_format=array_record pathmnist_specialist/
tfds build --overwrite --file_format=array_record dermamnist_specialist/

# tfds build --file_format=array_record --datasets=imagenet2012 --manual_dir=/data/giil/tensorflow_datasets/downloads/manual
