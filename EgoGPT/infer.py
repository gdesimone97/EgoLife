import debugpy
#debugpy.listen(("0.0.0.0", 5678))

import argparse
import copy
import os
import re
import sys
import warnings

from datasets import load_dataset, Video, concatenate_datasets
import numpy as np
import requests
import soundfile as sf
import torch
import torch.distributed as dist
import whisper
from decord import VideoReader, cpu, AudioReader
from collections import defaultdict
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.conversation import SeparatorStyle, conv_templates
from egogpt.mm_utils import get_model_name_from_path, process_images
from egogpt.model.builder import load_pretrained_model
from PIL import Image
from scipy.signal import resample
from pathlib import Path
import json

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def load_video(video_path=None, max_frames_num=16, fps=1):
    audio = AudioReader(video_path, sample_rate=16000)
    speech = audio[:].asnumpy()
    if speech.shape[0] > 1:
        speech = np.mean(speech, axis=1)
    speech = whisper.pad_or_trim(speech.astype(np.float32))
    speech: torch.Tensor = whisper.log_mel_spectrogram(speech, n_mels=128).squeeze(0).permute(1,0)
    speech_lengths = torch.LongTensor([speech.shape[0]])

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    if max_frames_num > 0 and len(frame_idx) > max_frames_num:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    return video, speech, speech_lengths


def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


def main(
    pretrained_path="checkpoints/EgoGPT-0.5b-Demo",
    video_path=None,
    audio_path=None,
    query="Please describe the video in detail.",
):
    warnings.filterwarnings("ignore")
    try:
        setup(0, 1)
    except ValueError:
        pass
    device = "cuda"
    device_map = "cuda"

    tokenizer, model, max_length = load_pretrained_model(
        pretrained_path, device_map=device_map
    )
    model.eval()

    conv_template = "qwen_1_5"
    question = f"<image>\n<speech>\n\n{query}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    video, speech, speech_lengths = load_video(
        video_path=video_path
    )
    speech = torch.stack([speech]).to(device).half()
    processor = model.get_vision_tower().image_processor
    processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values"]
    image = [(processed_video, video[0].size, "video")]

    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if part == "<image>":
            input_ids.append(IMAGE_TOKEN_INDEX)
        elif part == "<speech>":
            input_ids.append(SPEECH_TOKEN_INDEX)
        else:
            input_ids.extend(tokenizer(part).input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    image_tensor = [image[0][0].half()]
    image_sizes = [image[0][1]]
    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        speech=speech,
        speech_lengths=speech_lengths,
        do_sample=True,
        temperature=0.8,
        max_new_tokens=512,
        modalities=["video"],
        **generate_kwargs,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)
    return text_outputs[0]


if __name__ == "__main__":
    dataset_dir = "/home/dsmgpp000/code/MultiEngagement/AVDIAR_All/huggie_dataset"
    data = load_dataset(path=dataset_dir)
    data["train"] = concatenate_datasets([data["train"], data["validation"], data["test"]])
    data = data["train"]
    data = data.cast_column("video", Video(decode=False))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path", type=str, default="lmms-lab/EgoGPT-0.5b-Demo"
    )
    parser.add_argument(
        "--query", type=str, default="Please describe the video in detail."
    )
    args = parser.parse_args()
    df = defaultdict(lambda: defaultdict())
    for e in data:
        path = e["video"]["path"]
        text_outputs = main(args.pretrained_path, path, args.query)
        df[path]["desc"] = text_outputs
        
    with open("descriptions_AVDIAR_All.json", "w") as fil:
        json.dump(df, fil, indent=4)