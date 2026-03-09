#!/bin/bash

# tfds build cifar100_noisy/
# tfds build cifar100_longtail/

tfds build --overwrite --file_format=array_record cifar100_noisy/
tfds build --overwrite --file_format=array_record cifar100_longtail/

# tfds build --file_format=array_record --datasets=imagenet2012 --manual_dir=/data/giil/tensorflow_datasets/downloads/manual
