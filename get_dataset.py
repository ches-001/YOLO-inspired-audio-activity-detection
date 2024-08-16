import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
import argparse
import tqdm
import asyncio
import torchaudio
import math
import random
import glob
import shutil
import kaggle
from typing import List


def convert_audio_ext(audiofile: str, dest_ext: str="wav"):
    file_ext = audiofile.split(".")[-1]
    if file_ext == dest_ext:
        return
    audio_tensor, sample_rate = torchaudio.load(audiofile, backend="soundfile")
    torchaudio.save(
        audiofile.replace(f".{file_ext}", f".{dest_ext}"), 
        src=audio_tensor, 
        sample_rate=sample_rate
    )
    os.remove(audiofile)

async def convert_audio_ext_coro(audiofile: str, dest_ext: str, semaphore: asyncio.Semaphore):
    loop = asyncio.get_running_loop()
    async with semaphore:
        await loop.run_in_executor(None, lambda : convert_audio_ext(audiofile, dest_ext))


def glob_all_exts(dir: str, exts: List[str], recursive: bool=False):
    all_files = []
    for ext in exts:
        files = glob.glob(os.path.join(dir, "**", f"*.{ext}"), recursive=recursive)
        all_files += files
    return all_files
    

async def converter(dataset_dir: str, supported_exts: List[str], dest_ext: str="wav", num_concurrency: str=5):
    files = glob_all_exts(dataset_dir, supported_exts, recursive=True)
    if not files:
        return
    semaphore = asyncio.Semaphore(num_concurrency)
    tasks = []
    for file in files:
        tasks.append(asyncio.ensure_future(convert_audio_ext_coro(file, dest_ext, semaphore)))
    [await task for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":
    dataset_url = "https://www.kaggle.com/datasets/chinonsoonah/openbmat"
    dataset_name = "openbmat"
    supported_exts = ["mp3", "wav"]
    convert_to_ext = "wav"

    parser = argparse.ArgumentParser(description=f"Dataset Downloader")
    parser.add_argument(
        "--url", type=str, default=dataset_url, metavar="", 
        help=f"Kaggle URL to dataset (default = {dataset_url})"
    )
    parser.add_argument(
        "--name", type=str, default=dataset_name, metavar="", 
        help=f"Dataset name (default = {dataset_name})"
    )
    parser.add_argument(
        "--to_ext", type=str, default=convert_to_ext, metavar="", 
        help=f"Extension to convert dataset files to (default = {convert_to_ext})"
    )
    parser.add_argument(
        "--num_concurrency", type=int, default=4, metavar="", 
        help=f"Number of concurrent tasks for ext conversion (default = {4})"
    )
    args = parser.parse_args()

    dataset_url = args.url
    dataset_name = args.name
    convert_to_ext = args.to_ext

    dataset_dir = f"dataset/{dataset_name}"
    train_dir = f"{dataset_dir}/train"
    eval_dir = f"{dataset_dir}/eval"
    annotations_path = f"{dataset_dir}/annotations"

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    kaggle.api.dataset_download_cli(dataset_url.split("datasets/")[-1], path=dataset_dir, unzip=True)
    
    audio_files = glob_all_exts(dataset_dir, supported_exts, recursive=True)
    annotation_files = glob.glob(os.path.join(dataset_dir, "**", ".json"), recursive=True)

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
    
    valid_dirs = ["train", "eval", "annotations"]
    for d in os.listdir(dataset_dir):
        if d not in valid_dirs:
            shutil.rmtree(os.path.join(dataset_dir, d))

    asyncio.run(
        converter(
            dataset_dir, 
            supported_exts,
            dest_ext=convert_to_ext, 
            num_concurrency=args.num_concurrency
        )
    )