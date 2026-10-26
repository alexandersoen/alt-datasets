# Alternative Datasets

Alternative datasets based on "model cascade" experiments of "When does confidence-based cascade deferral suffice?."
Corruption settings are further expanded to some MedMNIST datasets.

Note: This repo only provide a minimal implementation of the datasets + their corruption. So I haven't properly defined anything except the builders themselves (no tests, no checksums, no `CITATIONS.bib`).

This package includes:

- `cifar100_labelnoise`: label noise applied to classes `0..n-1`
- `cifar100_longtail`: class-imbalance variants with head/tail truncation
- `cifar100_specialist`: specialist-biased variants with full specialist-class coverage and downsampled non-specialist classes
- `dermamnist`: based dataset wrapped for `tfds`.
- `pathmnist`: based dataset wrapped for `tfds`.
- `dermamnist_specialist`: specialist-biased variants with full specialist-class coverage and downsampled non-specialist classes
- `pathmnist_specialist`: specialist-biased variants with full specialist-class coverage and downsampled non-specialist classes
- `imagenet2012_dogs`: dog-specialist-biased variant with full specialist-class coverage and downsampled non-specialist classes

## Build instructions

Utilize the `tfds` CLI.

```bash
tfds build cifar100_labelnoise --file_format=array_record
```

Although optional, `array_record` is recommended. Allows for indexing, which makes things less painful if you need to switch to torch.

Can also use the bash file `build_all.sh` (the `imagenet` dataset is commented out).
