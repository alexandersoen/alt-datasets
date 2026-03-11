# CIFAR100 Alternative Datasets

Paper-specific CIFAR100 dataset variants used for cascade experiments.

This package includes:

- `cifar100_labelnoise`: label noise applied to classes `0..n-1`
- `cifar100_longtail`: class-imbalance variants with head/tail truncation

## Build instructions

Utilize the `tfds` CLI.

```bash
tfds build cifar100_labelnoise --file_format=array_record
```

Although optional, `array_record` is recommended. Allows for indexing, which makes things less painful if you need to switch to torch.

Can also use the bash file `build_all.sh`.
