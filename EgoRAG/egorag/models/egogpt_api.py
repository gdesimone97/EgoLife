import requests
from egorag.models.base import BaseQueryModel


class EgoGPTAPI(BaseQueryModel):
    def __init__(self, url):
        # import pdb; pdb.set_trace()

        self.url = url

    def inference_video(
        self,
        video_path,
        video_start_time,
        start_time,
        end_time,
        human_query,
        system_message=None,
    ):
        payload = {
            "prompt": human_query,
            "video_path": video_path,
            "max_frames_num": 32,
            "fps": 1,
            "video_start_time": video_start_time,
            "start_time": start_time,
            "end_time": end_time,
        }
        response = requests.post(self.url, json=payload)

        # 检查响应
        if response.status_code == 200:
            result = response.json()["generated_text"]
        else:
            raise Exception(
                f"API request failed with status code {response.status_code}"
            )
        return result[0]
