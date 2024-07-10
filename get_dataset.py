import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
import math
import random
import glob
import shutil
import kaggle


if __name__ == "__main__":
    dataset_url = "https://www.kaggle.com/datasets/chinonsoonah/openbmat"
    dataset_dir = "dataset"
    train_dir = "dataset/train"
    eval_dir = "dataset/eval"
    annotations_path = "dataset/annotations"
    dataset_name = dataset_url.split("datasets/")[-1]

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    kaggle.api.dataset_download_cli(dataset_name, path=dataset_dir, unzip=True)

    audio_files = glob.glob(os.path.join(dataset_dir, f"openBMAT/audio/*.wav"), recursive=False)
    annotation_files = glob.glob(os.path.join(dataset_dir, f"openBMAT/annotations/*.json"), recursive=False)

    n_samples = len(audio_files)
    train_data_size = math.ceil(0.8 * n_samples)
    eval_data_size = n_samples - train_data_size

    train_files = random.sample(audio_files, train_data_size)
    eval_files = [file for file in audio_files if file not in train_files]

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    for file in train_files: shutil.move(file, train_dir)

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    for file in eval_files: shutil.move(file, eval_dir)

    if not os.path.isdir(annotations_path):
        os.makedirs(annotations_path)
    for file in annotation_files: shutil.move(file, annotations_path)
    
    shutil.rmtree(os.path.join(dataset_dir, "openBMAT"))