import argparse
import pathlib
import tensorflow_datasets as tfds

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='Setup datasets to build')
    parser.add_argument('--manual-dir', type=str, help='Manual folder / dir path')

    return parser.parse_args()

def prepare_imagenet(manual_dir: pathlib.Path) -> None:
    print(f"Preparing ImageNet dataset from {manual_dir}")

    builder = tfds.builder('imagenet2012', file_format='array_record')
    dl_config = tfds.download.DownloadConfig(
        manual_dir=str(manual_dir)
    )
    builder.download_and_prepare(download_config=dl_config)
    print(builder.as_data_source())


if __name__ == '__main__':
    args = parse()

    manual_folder = pathlib.Path(args.manual_dir)
    prepare_imagenet(manual_folder)
