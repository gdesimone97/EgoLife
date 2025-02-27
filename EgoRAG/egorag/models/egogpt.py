try:
    from egogpt.model.builder import load_pretrained_model
    from egogpt.mm_utils import get_model_name_from_path, process_images
    from egogpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
    from egogpt.conversation import conv_templates, SeparatorStyle
except:
    print('Please install the omni-speech')
from PIL import Image
import requests
import copy
import torch
import numpy as np
import sys
import warnings
from decord import VideoReader, cpu
import soundfile as sf
from scipy.signal import resample
import whisper
import re
import os
import torch.distributed as dist
from moviepy.editor import VideoFileClip
from egorag.utils.util import time_to_frame_idx
import openai, json, re
from PIL import Image
from io import BytesIO
import base64
import decord
import numpy as np
from egorag.models.base import BaseQueryModel
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")
def split_text(text, keywords):
    # 创建一个正则表达式模式，将所有关键词用 | 连接，并使用捕获组
    pattern = '(' + '|'.join(map(re.escape, keywords)) + ')'
    # 使用 re.split 保留分隔符
    parts = re.split(pattern, text)
    # 去除空字符串
    parts = [part for part in parts if part]
    return parts


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


class EgoGPT(BaseQueryModel):
    def __init__(self, pretrained):
        setup(0, 1)
        self.device = 'cuda'
        device_map = 'cuda'
        self.tokenizer, self.model, self.max_length = load_pretrained_model(
            pretrained, device_map=device_map)

        self.model.eval()

    def process_video(self, video_path: str, video_start_time: int, start_time: int, end_time: int, fps=1):
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)

        # Get the actual FPS of the video
        video_fps = vr.get_avg_fps()

        # Convert time to frame index based on the actual video FPS
        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        print("start frame", start_frame)
        print("end frame", end_frame)

        # Ensure the end time does not exceed the total frame number
        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        # Adjust start_frame and end_frame based on video start time
        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame))) # 确保不会超过总帧数
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [i for i in range(start_frame, end_frame) if (i - start_frame) % int(video_fps / fps) == 0]

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()
        target_sr = 16000  # Set target sample rate to 16kHz
    
        # Load audio from video with resampling
        y, _ = librosa.load(video_path, sr=target_sr)
        
        # Convert time to audio samples (using 16kHz sample rate)
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        
        # Extract audio segment
        audio_segment = y[start_sample:end_sample]
        # Return processed video and the corresponding frame indices
        return {
        'processed_video': video,
        'frame_idx': frame_idx,
        'start_time': start_time,
        'end_time': end_time,
        'audio': audio_segment,
        'sample_rate': target_sr
    }


    def inference_video(self, video_path, video_start_time, start_time, end_time, human_query, system_message=None):

        processed_data = self.process_video(video_path,video_start_time,start_time,end_time)
        speech=processed_data['audio']
        video=processed_data['processed_video']
        frame_idx=processed_data['frame_idx']   
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
        speech=torch.stack([speech]).to("cuda").half()
        conv_template = 'qwen_1_5'  # Make sure you use correct chat template for different models
        question = '<image>\n' + '<speech>\n' + f'\n{human_query}'
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        processor = self.model.get_vision_tower().image_processor
        processed_video = processor.preprocess(
            video, return_tensors='pt')['pixel_values']
        image = [(processed_video, video[0].size, 'video')]
        parts = split_text(prompt_question, ['<image>', '<speech>'])
        input_ids = []
        for part in parts:
            if '<image>' == part:
                input_ids += [IMAGE_TOKEN_INDEX]
            elif '<speech>' == part:
                input_ids += [SPEECH_TOKEN_INDEX]
            else:
                input_ids += self.tokenizer(part).input_ids
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.long).unsqueeze(0).to(self.device)
        image_tensor = [image[0][0].half()]
        image_sizes = [image[0][1]]

        generate_kwargs = {'eos_token_id': self.tokenizer.eos_token_id}

        cont = self.model.generate(input_ids,
                                   images=image_tensor,
                                   image_sizes=image_sizes,
                                   speech=speech,
                                   speech_lengths=speech_lengths,
                                   do_sample=False,
                                   temperature=0.5,
                                   max_new_tokens=4096,
                                   modalities=['video'],
                                   **generate_kwargs)

        text_outputs = self.tokenizer.batch_decode(cont,
                                                   skip_special_tokens=True)
       
        return text_outputs[0]


    

    