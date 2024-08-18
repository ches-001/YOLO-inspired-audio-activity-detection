import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
import logging
import argparse
import tqdm
import asyncio
import torch
import torchaudio
import math
import random
import glob
import shutil
import kaggle
from typing import List

logger = logging.getLogger(__name__)

def convert_audio(audiofile: str, dest_ext: str="wav", target_sample_rate: int=22050):
    file_ext = audiofile.split(".")[-1]
    audio_tensor, sample_rate = torchaudio.load(audiofile, backend="soundfile")

    if sample_rate != target_sample_rate:
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to("cuda")
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, sample_rate, target_sample_rate
        ).cpu()

    if file_ext == dest_ext:
        return

    torchaudio.save(
        audiofile.replace(f".{file_ext}", f".{dest_ext}"), 
        src=audio_tensor, 
        sample_rate=sample_rate
    )
    os.remove(audiofile)

async def format_audio_coro(
        audiofile: str, 
        dest_ext: str, 
        target_sample_rate: int, 
        semaphore: asyncio.Semaphore
    ):
    loop = asyncio.get_running_loop()
    async with semaphore:
        await loop.run_in_executor(
            None, lambda : convert_audio(audiofile, dest_ext, target_sample_rate)
        )


def glob_all_exts(dir: str, exts: List[str], recursive: bool=False):
    all_files = []
    for ext in exts:
        files = glob.glob(os.path.join(dir, "**", f"*.{ext}"), recursive=recursive)
        all_files += files
    return all_files
    

async def format_and_resample(
        dataset_dir: str, 
        supported_exts: List[str], 
        target_sample_rate: int, 
        dest_ext: str="wav", 
        num_concurrency: str=5
    ):
    logger.info("converting audiofiles to their respective formats (extensions)...")
    files = glob_all_exts(dataset_dir, supported_exts, recursive=True)
    if not files:
        logger.info("No dataset found")
        return
    semaphore = asyncio.Semaphore(num_concurrency)
    tasks = []
    for file in files:
        tasks.append(asyncio.ensure_future(format_audio_coro(file, dest_ext, target_sample_rate, semaphore)))
    [await task for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":
    dataset_url = ""#"https://www.kaggle.com/datasets/chinonsoonah/openbmat"
    dataset_name = "openbmat"
    target_sample_rate = 22050
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
        "--target_sample_rate", type=int, default=target_sample_rate, metavar="", 
        help=f"sample rate to resample the audiofile to (default = {target_sample_rate})"
    )
    parser.add_argument(
        "--num_concurrency", type=int, default=4, metavar="", 
        help=f"Number of concurrent tasks for ext conversion (default = {4})"
    )
    parser.add_argument("--format_only", action="store_true", 
        help="If set, dataset in the folder specified by the filename are simply formatted"
    )
    args = parser.parse_args()

    dataset_url = args.url
    dataset_name = args.name
    convert_to_ext = args.to_ext
    target_sample_rate = args.target_sample_rate
    dataset_dir = f"dataset/{dataset_name}"
    train_dir = f"{dataset_dir}/train"
    eval_dir = f"{dataset_dir}/eval"
    annotations_path = f"{dataset_dir}/annotations"

    if not args.format_only:
        if len(dataset_url) > 0:
            logger.info(f"Downloading {dataset_url}...")
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
            kaggle.api.dataset_download_cli(dataset_url.split("datasets/")[-1], path=dataset_dir, unzip=True)
        
        if not os.path.exists(dataset_dir):
            raise OSError(f"path: {dataset_dir} does not exist")
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
        format_and_resample(
            dataset_dir,
            supported_exts,
            target_sample_rate,
            dest_ext=convert_to_ext, 
            num_concurrency=args.num_concurrency
        )
    )