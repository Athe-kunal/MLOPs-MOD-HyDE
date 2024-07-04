import torch
from transformers import pipeline
import os
from config import *
from tqdm import tqdm
import json
import wandb
import re
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(),override=True)
assert (
    int(torch.__version__[0]) >= 2
), "Please ensure that the torch version is above 2 to use better transformers"

WHISPHER_BATCH_SIZE = 1
WHISPHER_CHUNK_LENGTH = 100
def generate_transcripts(output_folder: str, idx: int = 0):
    wandb.login(key=os.environ['WANDB_API_KEY'])
    run = wandb.init(
        project=os.environ['WANDB_PROJECT'],
        config={"source": "YouTube Video", "data": output_folder},
        entity=os.environ['WANDB_ENTITY'],
        group=output_folder,
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    artifact = wandb.Artifact(name=output_folder + "_data", type="dataset")
    json_folder_name = output_folder + "_TRANSCRIPTS"
    os.makedirs(json_folder_name, exist_ok=True)
    for audio_files in tqdm(os.listdir(output_folder)[idx:]):
        print(f"Generating transcripts for {audio_files}")
        title = audio_files.split(".mp3")[0]
        relative_audio_path_mp3 = os.path.join(output_folder, audio_files)
        relative_audio_path_json = json_folder_name + "/" + title + ".json"
        if os.path.exists(relative_audio_path_json):
            print(
                f"Transcripts are already generated for {title} in the folder {output_folder}, please check"
            )
            continue
        outputs = pipe(
            relative_audio_path_mp3,
            chunk_length_s=WHISPHER_CHUNK_LENGTH,
            batch_size=WHISPHER_BATCH_SIZE,
            return_timestamps=True,
        )
        with open(relative_audio_path_json, "w") as fp:
            json.dump(outputs, fp)
        torch.cuda.empty_cache()
    artifact.add_dir(local_path=json_folder_name, name="TRANSCRIPTS")
    # artifact.add_dir(local_path=output_folder,name="AUDIO")
    run.log_artifact(artifact)
    del pipe
    torch.cuda.empty_cache()