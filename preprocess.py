import argparse

from utils.tools import get_configs_of
from preprocessor import ljspeech, vctk


def main(preprocess_config, model_config, train_config):
    if "LJSpeech" in preprocess_config["dataset"]:
        preprocessor = ljspeech.Preprocessor(preprocess_config, model_config, train_config)
    if "VCTK" in preprocess_config["dataset"]:
        preprocessor = vctk.Preprocessor(preprocess_config, model_config, train_config)

    preprocessor.build_from_path()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    main(preprocess_config, model_config, train_config)
