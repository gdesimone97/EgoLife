import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from egorag.database.Chroma import Chroma
from egorag.models import BaseQueryModel
from tqdm import tqdm


def get_ids(question, results):
    system_prompt = """
    You are an AI assistant tasked with helping to process and analyze search results for specific questions. 
    When a user provides a question along with a set of search results, your job is to:

    1. Review all search results.
    2. Find the most recent entry that is directly related to the user's question.
    3. Consider both the relevance of the document and the time of the event (based on the provided timestamp).
    4. Return the `id` of the most relevant and latest document.
    5. Output the id with format 'ID: <id>'

    Please ensure that you consider both the content of the document and the timestamp to identify the most recent and relevant document.

    Your responses should only include the format `ID: <id>` of the selected document.

    Here is an example to guide your process:

    **Question:**  
    "When was the last time the guitar was played?"

    **Search Results:**  
    [
        {'id': 'DAY2-16170000-16173000_5', 'document': ['Katrina mentioned that the projector on the second floor was nice too and asked if we could play some music.', 'I agreed, "Play some music.', '" Then I placed the box Lucia handed me onto the desk, opened it, looked at Lucia, and then closed the box.'], 'date': 2, 'end_time': 16173000, 'distance': 0.47747164964675903}, 
        {'id': 'DAY2-18163000-18170000_2', 'document': ['As we talked, he scratched his nose, which made me chuckle.', 'We were having a lively conversation, and he even started playing the guitar.', "I mentioned, 'Ideally, we should have a script supervisor, it would make the work much easier."], 'date': 2, 'end_time': 18170000, 'distance': 0.5553568601608276}, 
        {'id': 'DAY2-18200000-18203000_3', 'document': ["I said, 'The An and Ankang fish are collected.", "' Then I tilted my head and continued chatting with Choiszt, who was strumming his guitar while responding to me.", "I said, '20 minutes, 20 minutes down,' and added, 'That's fine, one more hour, right?' At that moment, I noticed Nicous beside me, touching his hair."], 'date': 2, 'end_time': 18203000, 'distance': 0.616065502166748}, 
        {'id': 'DAY2-18270000-18273000_4', 'document': ['Sitting on the bed with my legs bent, I glanced at my colleagues.', 'Some were busy at their computers, a boy was playing the guitar, and others were focused on their tasks.', 'I observed the scattered electronics and messy bedding, contemplating whether to join their discussion.'], 'date': 2, 'end_time': 18273000, 'distance': 0.6114476323127747}, 
        {'id': 'DAY3-16163000-16170000_5', 'document': ["I put down the box and bottle in my hands and hesitantly said, 'Seems like not.", "Yeah, then I'll start playing.", "' The constantly changing scores on the screen confused me, so I asked, 'What does that mean?' At that moment, Tasha asked, 'Can't we play together?' I replied, 'Sure, you go first then."], 'date': 3, 'end_time': 16170000, 'distance': 0.6253791451454163}, 
        {'id': 'DAY3-21160000-21163000_3', 'document': ["We were chatting happily, and Lucia laughed, saying, 'They might just be simply moving the dishes from one place to another.", "' I joked, 'Playing music, hahaha, continue.", "' Then I said, 'Alright, alright, even though we can't hit it, keep playing the music."], 'date': 3, 'end_time': 21163000, 'distance': 0.604546308517456}, 
        {'id': 'DAY3-22053000-22060000_3', 'document': ["Tasha asked, 'Hey, didn't you play that the other day?' to which I replied, 'Yeah, it's good.", 'You gotta play well, gotta play.', "' Just then, I saw Tasha and Katrina coming over, dressed in casual clothes with happy smiles on their faces."], 'date': 3, 'end_time': 22060000, 'distance': 0.5654604434967041}, 
        {'id': 'DAY3-22453000-22460000_3', 'document': ['Lucia also came over, and Shure was looking into my eyes.', 'I moved closer to see him as he played guitar and sang.', 'I heard him sing, "I really hope you can muster up some courage.'], 'date': 3, 'end_time': 22460000, 'distance': 0.49885010719299316}, 
        {'id': 'DAY3-22460000-22463000_4', 'document': ["' I noticed the table was filled with food and drinks, and everyone seemed very relaxed.", "Shure was playing the guitar, and I nodded my head to the rhythm; his music was so beautiful and moving that I couldn't help but sway gently.", 'Lucia and Tasha were also eating at the table, while Alice continued to grill food, with the room dimly lit and the grill emitting fragrant smoke.'], 'date': 3, 'end_time': 22463000, 'distance': 0.6172605156898499}, 
        {'id': 'DAY3-22463000-22470000_0', 'document': ['I saw Shure playing the guitar and nodded to the rhythm of the music; the surrounding environment was very quiet.', 'Shure was playing happily, and the atmosphere was relaxed.'], 'date': 3, 'end_time': 22470000, 'distance': 0.6303433179855347}
    ]

    **Explanation of Fields in Search Results:**

    - **id**: This represents the time when the event described in the document occurred. The format is `DAYx_HHMMSSFF_HHMMSSFF`, where the first timestamp is the start time and the second is the end time of the event.
    - **document**: This is the textual description of the event that happened. It tells you what occurred at the given time.
    - **distance**: This indicates how similar the document is to the search query. The smaller the distance, the higher the relevance or similarity between the query and the document.
    - **date**: This represents the day on which the event described in the document occurred. It is a single integer (e.g., 1, 2, 3, etc.), indicating the specific day (e.g., Day 1, Day 2, etc.) relative to a defined starting point or timeline. Unlike end_time, which includes a timestamp, date only specifies the day.
    - **end_time**: This is the timestamp representing the moment when the event described in the document ended. It is also in the format `HHMMSSFF`.

    All times are formatted as `HHMMSSFF`, where:
    - `HH` stands for hours (00-23),
    - `MM` stands for minutes (00-59),
    - `SS` stands for seconds (00-59),
    - `FF` represents the frame (00-20).

    Note:The search results are already arranged in chronological order, from the earliest to the latest.
    
    **Expected Output:**  
    "ID: DAY3-22463000-22470000_0"

    Explanation: The most recent action related to the guitar being played is recorded at `DAY3_22463000_22470000_0`, which occurred at `22463000` and directly answers the user's question about the last time the guitar was tuned or played.
    """

    prompt = f"Question:{question}\n Search Results:{results}"
    answer = call_gpt4(prompt=prompt, system_message=system_prompt, temperature=0.1)
    print(answer)
    # Extract the ID from the answer using regex
    id_match = re.search(r"ID:\s*(DAY\d+-\d{8}-\d{8}_\d)", answer)
    if id_match:
        id = id_match.group(1)

    else:
        print("No ID found in the response.")
        id = None
    return id


def get_max_endtime_result(raw_results):
    # 首先找出最大的date
    max_date = max(metadata["date"] for metadata in raw_results["metadatas"][0])

    # 在最大date的条目中找出最大的end_time
    max_end_time_idx = max(
        (
            i
            for i, metadata in enumerate(raw_results["metadatas"][0])
            if metadata["date"] == max_date
        ),
        key=lambda i: raw_results["metadatas"][0][i]["end_time"],
    )

    # 创建一个新字典，只包含选中的数据
    return {
        "ids": [raw_results["ids"][0][max_end_time_idx]],
        "documents": [raw_results["documents"][0][max_end_time_idx]],
        "metadatas": [raw_results["metadatas"][0][max_end_time_idx]],
        "distances": [raw_results["distances"][0][max_end_time_idx]],
    }


def timestamp_operation(timestamp1, timestamp2, operation="add", fps=30):
    """
    Perform addition or subtraction on timestamps.

    Args:
        timestamp1: First timestamp (int or str, 7 or 8 digits)
        timestamp2: Second timestamp (int or str, 7 or 8 digits)
        operation: 'add' or 'subtract' (default: 'add')
        fps: Frames per second (default: 30)

    Returns:
        int: Result of timestamp operation
    """
    # Convert to strings and pad to 8 digits if needed
    ts1 = str(timestamp1).zfill(8)
    ts2 = str(timestamp2).zfill(8)

    # Extract components
    h1, m1, s1, f1 = int(ts1[:2]), int(ts1[2:4]), int(ts1[4:6]), int(ts1[6:])
    h2, m2, s2, f2 = int(ts2[:2]), int(ts2[2:4]), int(ts2[4:6]), int(ts2[6:])

    # Convert all to frames using fps
    total1 = ((h1 * 60 + m1) * 60 + s1) * fps + f1
    total2 = ((h2 * 60 + m2) * 60 + s2) * fps + f2

    # Perform operation
    if operation == "add":
        result = total1 + total2
    else:  # subtract
        result = total1 - total2
        if result < 0:
            result = 0

    # Convert back to HHMMSSFF format
    frames = result % fps
    seconds = (result // fps) % 60
    minutes = (result // (fps * 60)) % 60
    hours = (result // (fps * 60 * 60)) % 100

    # Scale frames to 0-99 range
    frames = int((frames / fps) * 100)

    return int(f"{hours:02d}{minutes:02d}{seconds:02d}{frames:02d}")


def parse_video_map(video_map):
    parsed_data = []
    for key, video_path in video_map.items():
        # 分割 key 获取 date, start_time, end_time
        key_parts = key.split("-")
        date = key_parts[0]
        start_time = int(key_parts[1])
        end_time = int(key_parts[2])

        # 提取 video_path 中的时间部分
        match = re.search(r"_(\d{8})\.mp4$", video_path)  # 提取视频路径中的时间部分
        if match:
            video_start_time = int(match.group(1))  # 将提取到的时间转为整数
        else:
            video_start_time = None  # 如果未找到时间部分，则设为 None

        # 提取 date 中的数字部分作为 int_date
        match = re.search(r"\d+", date)
        int_date = int(match.group()) if match else None

        # 将解析后的数据添加到结果列表中
        parsed_data.append(
            {
                "date": date,
                "int_date": int_date,
                "start_time": start_time,
                "end_time": end_time,
                "video_path": video_path,
                "video_id": key,
                "video_start_time": video_start_time,  # 添加视频的开始时间
            }
        )
    return parsed_data


def transform_timedict(time_dict):
    # 使用 defaultdict 来自动创建列表
    date_time_mapping = defaultdict(list)

    # 填充字典
    for entry in time_dict:
        date_time_mapping[entry["date"]].append(str(entry["time"]))

    # 对每个日期下的时间进行排序
    for date in date_time_mapping:
        date_time_mapping[date] = sorted(date_time_mapping[date])
    return date_time_mapping


import time
from typing import Optional

import requests


def call_gpt4(
    prompt: str,
    system_message: str = "You are an effective first perspective assistant.",
    temperature=0.9,
    top_p=0.95,
) -> Optional[str]:
    """
    Call GPT-4 API with given prompt and system message.

    Args:
        prompt (str): The user prompt to send to GPT-4
        system_message (str): System message to set context for GPT-4

    Returns:
        Optional[str]: The response content from GPT-4, or None if request fails after retries
    """
    # Configuration
    GPT4V_KEY = os.getenv("GPT4V_KEY")
    GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT")

    if GPT4V_KEY is None or GPT4V_ENDPOINT is None:
        raise ValueError("GPT4V_KEY Or GPT4V_ENDPOINT IS NOT SETTING")

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 2200,
    }

    retries = 5
    for attempt in range(retries):
        try:
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Error: {e}")
            if attempt < retries - 1:  # No delay needed after the last attempt
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print("All retry attempts failed.")
                return "error"


class RagAgent(ABC):
    def __init__(
        self,
        model: Optional[BaseQueryModel] = None,
        database_t: Optional[Chroma] = None,
        database_i: Optional[Chroma] = None,
        name: str = "NULL",
        video_base_dir: str = "data/videos",
    ):
        super().__init__()
        self.model = model
        self.database_t = database_t
        self.database_i = database_i
        self.name = name
        self.video_base_dir = video_base_dir

    def calculate_accuracy(self, answers, save_to_file=None):
        total_count = 0
        correct_count = 0
        results = []

        for answer in answers:
            # 获取正确答案和模型回答
            correct_answer = answer["metadata"]["answer"]
            model_answer = answer["model_option"]

            # 模型回答有效且不为 None 时，才进行比较
            total_count += 1
            if model_answer is None or not model_answer.strip():
                # 模型回答为空或None，视为错误
                is_correct = False
            else:
                # 如果模型的回答与正确答案匹配，增加正确数
                is_correct = model_answer.strip() == correct_answer.strip()

            correct_count += is_correct

            # 记录当前回答的结果
            answer["is_correct"] = is_correct
            results.append(answer)
        # 计算正确率，如果总题数为0，返回0避免除以0的错误
        accuracy = correct_count / total_count if total_count > 0 else 0

        # 如果传入了保存文件的地址，则将结果保存为 JSON 文件
        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"accuracy": accuracy, "results": results},
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        # 返回正确率
        return accuracy

    def get_video_mapping(self, sorted_video_times, all_times):
        video_path_mapping = {}
        for day in sorted_video_times.keys():
            date = day
            base_dir = self.video_base_dir
            video_time_list = sorted_video_times[date]
            all_time_list = all_times[date]
            for i in range(len(all_time_list) - 1):
                start_time = all_time_list[i]
                end_time = all_time_list[i + 1]

                # 找到 start_time 所属的 sorted_video_time 段
                for j in range(len(video_time_list) - 1):
                    if video_time_list[j] <= start_time < video_time_list[j + 1]:
                        # 确定视频地址，以 sorted_video_time[j] 为文件名
                        video_path = os.path.join(
                            base_dir,
                            f"{date}",
                            f"{date}_{self.name}_{video_time_list[j]}.mp4",
                        )
                        break
                else:
                    # 如果未找到合适的段（理论上不应该发生，如果 merged_times 与 sorted_video_time 对应）
                    video_path = None

                # 将时间段和对应的视频路径添加到字典中
                video_path_mapping[f"{date}-{start_time}-{end_time}"] = video_path
        return video_path_mapping

    def add_to_db_t(self, id, sample, caption_args, human_query, system_message=None):
        time_start = time.time()
        start_time = sample["start_time"]
        end_time = sample["end_time"]
        video_path = sample["video_path"]
        date = sample["int_date"]
        video_start_time = sample["video_start_time"]

        try:
            # Set timeout to 30 seconds
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Inference took too long")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)

            caption = self.model.inference_video(
                video_path, video_start_time, start_time, end_time, human_query
            )
            signal.alarm(0)  # Disable alarm
            print(caption)
        except TimeoutError as te:
            print(f"Timeout error in captioning: {te}")
            return None
        except Exception as e:
            print(f"Error in captioning: {e}")
            return None

        if caption is None or caption == "" or caption == "Error in captioning":
            return None

        sentences = [
            sentence.strip() + "."
            for sentence in caption.split(".")
            if sentence.strip()
        ]
        embeddings = self.database_t.embedding_function(sentences)
        # 为每个句子分配一个新的 id
        new_ids = [f"{id}_{i}" for i in range(len(sentences))]
        # 创建新的 add_element
        add_element = dict(
            ids=new_ids,
            documents=sentences,
            embeddings=embeddings,
            metadatas=[
                {
                    "start_time": int(start_time),
                    "end_time": int(end_time),
                    "video_path": video_path,
                    "date": date,
                }
            ]
            * len(sentences),
        )  # 复制 metadata 以匹配每个子句

        collection = self.database_t.collection
        collection.add(**add_element)

        print(f"Database_t: Segment {id} processed and added to the collection.")
        print(
            f"Current collection size: {len(collection.get(include=['documents'])['ids'])}"
        )
        print(f"Time cost: {time.time()-time_start:.2f}s")

        # 返回处理后的结果
        return [
            {"id": new_id, "caption": sentence}
            for new_id, sentence in zip(new_ids, sentences)
        ]

    def rag_CLIP_text(self, human_query, n_results=2):
        # similarity between user_query and captions
        results = self.database_t.collection.query(
            query_texts=[human_query], n_results=n_results
        )

        return results["ids"][0]

    def create_database_from_query(
        self,
        video_paths,
        query_json,
        human_query,
        caption_args,
        system_message=None,
        rag="rag_CLIP_t",
    ):
        video_time = [
            {
                "date": re.search(r"DAY\d", path).group(0),  # 提取日期，如DAY1, DAY2
                "time": re.search(r"_(\d{8})\.mp4", path).group(1),  # 提取时间，格式为8位数字
            }
            for path in video_paths
            if re.search(r"DAY\d", path) and re.search(r"_(\d{8})\.mp4", path)
        ]

        sorted_video_time = sorted(video_time, key=lambda x: (x["date"], x["time"]))
        query_time = []
        seen_times = set()
        for data in query_json:
            date = data["query_time"]["date"]
            time = data["query_time"]["time"]
            if (date, time) not in seen_times:
                query_time.append({"date": date, "time": time})
                seen_times.add((date, time))
        sorted_query_time = sorted(query_time, key=lambda x: (x["date"], x["time"]))

        # 合并并再次按日期和时间排序
        all_times = sorted(
            sorted_video_time + sorted_query_time, key=lambda x: (x["date"], x["time"])
        )

        sorted_video_time = transform_timedict(sorted_video_time)
        all_times = transform_timedict(all_times)
        video_mapping = self.get_video_mapping(sorted_video_time, all_times)
        samples = parse_video_map(video_mapping)

        if rag in ["rag_CLIP_t"]:
            ids = self.database_t.collection.get()["ids"]
            existing_video_ids = set(id.rsplit("_", 1)[0] for id in ids)
            for id, sample in tqdm(
                enumerate(samples), total=len(samples), desc="Generating caption"
            ):
                video_id = sample["video_id"]
                if video_id in existing_video_ids:
                    print(f"Already in database: {video_id}, skip.")
                    continue

                self.add_to_db_t(
                    video_id, sample, caption_args, human_query, system_message
                )

    def create_database_from_json(self, json_path):
        """Creates a database from a JSON file containing caption data.c

        Args:
            json_path (str): Path to the JSON file
        """
        # Load JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries from {json_path}")

        # Process each entry
        for idx, entry in tqdm(
            enumerate(data), total=len(data), desc="Processing entries"
        ):
            try:
                # Extract metadata
                date_match = re.search(r"DAY(\d+)", entry["date"])
                int_date = int(date_match.group(1)) if date_match else 0
                date = entry["date"]
                video_name = f'{date}_{self.name}_{entry["start_time"]}.mp4'
                video_path = entry.get(
                    "video_path",
                    os.path.join(self.video_base_dir, self.name, date, video_name),
                )
                # Handle ellipsis and proper sentence splitting

                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s*", entry["text"])
                    if sentence.strip()
                ]
                # Create entry ID in format DAYX_STARTTIME_ENDTIME
                entry_id = f"{entry['date']}-{entry['start_time']}-{entry['end_time']}"
                new_ids = [f"{entry_id}_{i}" for i in range(len(sentences))]

                # Check if any of the IDs already exist
                existing_docs = self.database_t.collection.get(
                    ids=new_ids, include=["documents"]
                )
                if existing_docs["ids"] and any(existing_docs["documents"]):
                    print(f"IDs {entry_id} already exist with content, skipping.")
                    continue

                # Create metadata
                metadata = {
                    "start_time": int(entry["start_time"]),
                    "end_time": int(entry["end_time"]),
                    "date": int_date,
                    "video_path": video_path,
                }

                # Get embeddings using database's embedding function

                embeddings = self.database_t.embedding_function(sentences)
                # Create add_element dictionary

                add_element = {
                    "ids": new_ids,
                    "documents": sentences,
                    "metadatas": [metadata] * len(sentences),
                    "embeddings": embeddings,
                }

                # Add to collection
                self.database_t.collection.add(**add_element)
                if idx % 100 == 0:  # Print progress every 100 entries
                    print(
                        f"Added {idx} entries. Current collection size: {len(self.database_t.collection.get(include=['documents'])['ids'])}"
                    )
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                continue

        print("Database creation completed!")
        print(
            f"Final collection size: {len(self.database_t.collection.get(include=['documents'])['ids'])}"
        )

    def generate_event_data(self, q_date, q_start_time, q_end_time):
        docs_to_use = self.database_t.collection.get(
            where={
                "$and": [
                    {"date": {"$eq": q_date}},
                    {"start_time": {"$gte": q_start_time}},
                    {"end_time": {"$lte": q_end_time}},
                ]
            },
            include=["documents"],
        )

        context = docs_to_use["documents"]

        response = call_gpt4(
            f"You will be provided with some descriptions. Merge events into one single event based on these descriptions. Do not include uncertain information, speculation, or divergent content. Do not describe the atmosphere or emotions. Dismiss those provided content that are abstract or ambiguous. If the descriptions mention names of people who interacted with 'me', make sure to retain this information. Directly provide the summarized main events without adding any additional remarks or explanations. All descriptions: {context}"
        )

        if response:
            print(f"Generate unaligned event diary.")
        else:
            print(f"Error processing unaligned event diary.")
            response = ""

        return response

    def filter_event_data(self, date, time, hour_docs, date_docs):
        date = int(date[-1])
        time = int(time)
        date_data = [entry for entry in date_docs if entry["date"] < date]
        hour_data = [
            entry
            for entry in hour_docs
            if entry["date"] == date and entry["end_time"] <= time
        ]
        if hour_data == []:
            last_event = {}
            last_event["generated_text"] = self.generate_event_data(
                date, hour_docs[0]["start_time"], time
            )
            last_event["date"] = date
            last_event["start_time"] = hour_docs[0]["start_time"]
            last_event["end_time"] = time
            hour_data.append(last_event)

        elif hour_data[-1]["end_time"] < time:
            last_event = {}
            last_event["generated_text"] = self.generate_event_data(
                date, hour_data[-1]["end_time"], time
            )
            last_event["date"] = date
            last_event["start_time"] = hour_data[-1]["end_time"]
            last_event["end_time"] = time
            hour_data.append(last_event)

        all_date_data = date_data + hour_data
        return all_date_data, hour_data

    def process_query_results(self, raw_results):
        """
        Process raw query results into a list of individual results with specific fields.

        Args:
            raw_results (dict): Raw results dictionary from the database query

        Returns:
            list: List of dictionaries containing processed results
        """
        processed_results = []

        # Check if we have results
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return processed_results

        # Get the first (and usually only) batch of results
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        # Combine results into a list of tuples for sorting
        combined_results = [
            (
                ids[i],
                documents[i],
                metadatas[i]["date"],
                metadatas[i]["end_time"],
                distances[i],
            )
            for i in range(len(ids))
        ]

        # Sort results first by date and then by end_time
        combined_results.sort(
            key=lambda x: (x[2], x[3])
        )  # Sort by date and then by end_timeS
        # Process each sorted result
        for id, document, date, end_time, distance in combined_results:
            results = self.database_t.get_caption(id=id, n_result=1)
            expand_documents = results["documents"]
            processed_result = {
                "id": id,
                "document": expand_documents,
                "date": date,
                "end_time": end_time,
                "distance": distance,
            }
            processed_results.append(processed_result)

        return processed_results

    def get_range(self, question, date, time, hour_docs, date_docs):
        int_date = int(date[-1])
        all_date, hour = self.filter_event_data(date, time, hour_docs, date_docs)

        system_message_l1 = f"""You will be provided with a JSON dataset containing summaries of events spanning multiple days.

        Given the question: **{question}**, your task is to determine the most relevant date from the dataset that can be used to answer the question.

        ### Considerations:
        - The reference time for the question is **{date}, {time}**.
        - You must carefully analyze the necessary time periods to derive an accurate response.
        - Apply logical reasoning and general knowledge to ensure an appropriate selection.
        - **Select only one date** that is most likely to contain the required information.
        - If the question contains keywords like "yesterday", "last night", "today", etc., you should make the selection based on the reference date.

        ### Output Format (Strictly Adhere):
        - Your response **must** follow this exact format: [X], where X represents the day number.
        - Example: [3] for DAY3.
        - **Do not** include any additional text, explanations, or formatting beyond the required output.
        """

        prompt_l1 = f"all json {all_date}"
        response_l1 = call_gpt4(
            prompt_l1, system_message=system_message_l1, temperature=0.8
        )
        print(response_l1)
        # Extract number from [X] format
        match = re.match(r"\[(\d+)\]", response_l1.strip())
        if match:
            target_date = int(match.group(1))
        else:
            target_date = int_date
        print(target_date)
        if target_date == int_date:
            hours_data = hour
        else:
            hours_data = [entry for entry in hour_docs if entry["date"] == target_date]

        system_message_l2 = f"""I will provide you with event descriptions, including their start time and end time in the format HHMMSSFF.
        For the question "{question}", determine the most appropriate time range that can be used to answer the question.
        The question time is {date}, {time}.

        You must carefully analyze the time periods necessary to derive an answer, incorporating logical reasoning and commonsense understanding.
        For each question, follow these thinking steps:
        - Identify the key elements from the question.
        - Infer a relevant event based on the key elements.
        - Determine the most probable time range associated with the event.

        Examples:
        Q1: "What is the first ingredient in the shopping cart yesterday?"
        - You should first find the key elements: shopping cart, first ingredient.
        - Then you might conclude the relevant event: shopping activity.
        - Finally, you should determine the time range: likely during the shopping event.

        Q2: "When was the pizza placed on the table?"
        - You should first find the key elements: pizza, placed on the table.
        - Then you might conclude the relevant event: meal preparation or serving.
        - Finally, you should determine the time range: likely within the meal preparation or serving period.

        Q3: "Where were the flowers placed before?"
        - You should first find the key elements: flowers, placed before.
        - Then you might conclude the relevant event: movement of the flowers.
        - Finally, you should determine the time range: likely before the flowers were relocated.

        Now you understand how to think and determine the time range. But your response should provide a SINGLE time range, including both the start timestamp and end timestamp.
        Ensure that your output contains only the time range in the following format:
        [Start Time]
        [End Time]

        Example output:
        [11000000]
        [14000000]
        """
        prompt_l2 = f"all description: {hours_data}"
        response_l2 = call_gpt4(
            prompt_l2, system_message=system_message_l2, temperature=0.8
        )
        start_time_match = re.search(r"\[(\d{8})\]", response_l2.split("\n")[0])
        end_time_match = re.search(r"\[(\d{8})\]", response_l2.split("\n")[1])

        start_time = (
            int(start_time_match.group(1))
            if start_time_match
            else hours_data[0]["start_time"]
        )
        end_time = (
            int(end_time_match.group(1))
            if end_time_match
            else hours_data[-1]["end_time"]
        )
        print(start_time)
        print(end_time)
        return target_date, start_time, end_time

    def query(self, query_dict, event_data, date_docs):
        question, query, formatted_options, time, date = self.parse_query(query_dict)
        if "last time" in question.lower() or "the last" in question.lower():
            print("handle the last time type question")

            system_message = """
                You are an intelligent assistant designed to:
                1. Extract concise and relevant keywords from user-provided questions.
                2. Determine the search time range based on the options provided (if applicable).

                ## Task ##
                - Analyze the question and extract the most critical phrases or terms that capture the essence of the query.
                - If the question involves time-related options, calculate the latest time mentioned in the options and provide the search time range accordingly.
                - If the question does not involve time-related options, set the search time range to the question time minus 5 minutes.

                ## Keywords Extract Rules ##
                1. Keywords should be as short as possible and directly tied to the question's context, only focus on action and items.
                
                2. Use ONLY verb + direct noun (remove locations, adjectives, and extra details).

                3. If specific NAME in the question, the keywords MUST contain NAMES.
                
                ## Time Calculation Rules ##
                1. Time format is always HHMMSSFF.
                
                2. For time-related questions:
                - Identify the latest time mentioned in the options.
                - Set the search time range to that time BASED ON THE QUESTION TIME.
                
                
                3. For non-time-related questions:
                - Set the search time range to the question time minus 5 minutes.
                
                5. The search time range MUST ALWAYS PRECEDE the question time.
                
                6. Please decide the Search Time Range Carefully!!!!
                
                
                ## Output Format ##
                Keywords: [extracted_keyword_here]
                Search Time Range: [search_time_range]

                ## Examples ##
                1. Q: What did we end up eating the last time we ordered takeout?
                Options:
                A. "KFC",
                B. "Noodles",
                C. "Dumplings",
                D. "Pizza",
                Question Time: DAY1 18304214
                Output: 
                Keywords: [ordered takeout]
                Search Time Range: [DAY1 18250000]

                2. Q: When was the last time items were taken from the refrigerator?
                Options:
                A. "Last night",
                B. "Yesterday at noon",
                C. "This morning",
                D. "Yesterday morning",
                Question Time: DAY2 10443918
                Output: 
                Keywords: [take out from refrigerator]
                Search Time Range: [DAY2 09000000]

                3. Q: When was the last time I received a receipt?
                Options:
                A. "The day before yesterday afternoon",
                B. "The day before yesterday at noon",
                C. "Yesterday afternoon",
                D. "Last night",
                Question Time: DAY5 16353500
                Output:
                Keywords: [receive receipt]
                Search Time Range: [DAY4 23590000]

                4. Q: What food was the microwave last used to heat?
                Options:
                A. "Pizza",
                B. "Soup",
                C. "Rice",
                D. "Noodles",
                Question Time: DAY3 12004512
                Output:
                Keywords: [use microwave]
                Search Time Range: [DAY3 11550000]
                """
            prompt = f"Question: {question} \nOptions: \n{formatted_options}\nQuestion Time: {date} {time}\nOutput:"

            keyword_raw = call_gpt4(
                prompt=prompt, system_message=system_message, temperature=0.1
            )
            # Extract keywords and time range using regex
            keyword_pattern = r"Keywords:\s*\[([^\]]+)\]"
            time_range_pattern = r"Search Time Range:\s*\[([^\]]+)\]"

            # Extract keywords
            keyword_match = re.search(keyword_pattern, keyword_raw)
            keywords = keyword_match.group(1).strip() if keyword_match else question
            # Extract time range
            time_range_match = re.search(time_range_pattern, keyword_raw)
            if time_range_match:
                search_time = time_range_match.group(1).strip()
                # Split into day and time if the format is "DAYX XXXXXXXX"
                search_time_parts = search_time.split()
                search_date = (
                    int(search_time_parts[0].replace("DAY", ""))
                    if len(search_time_parts) > 1
                    else int(date[-1])
                )
                search_timestamp = (
                    int(search_time_parts[1])
                    if len(search_time_parts) > 1
                    else int(search_time)
                )
            else:
                search_date = int(date[-1])
                search_timestamp = int(time)
            print(keywords)
            print(search_timestamp)
            raw_results = self.database_t.custom_query(
                query_texts=[keywords],
                n_results=50,
                where={
                    "$or": [
                        {"date": {"$lt": search_date}},
                        {
                            "$and": [
                                {"end_time": {"$lte": search_timestamp}},
                                {"date": {"$eq": search_date}},
                            ]
                        },
                    ]
                },
                filter_first=False,
            )
            all_query_results = self.process_query_results(raw_results)
            print(all_query_results)
            filter_id = get_ids(question=question, results=all_query_results)
            results = self.database_t.get_caption(id=filter_id, n_result=1)
            best_result = results["item"]
            # last_result=get_max_endtime_result(raw_results)
            query_result = [
                {
                    "query_range": None,
                    "docs": best_result,
                    "start_time": None,
                    "end_time": None,
                    "date": None,
                    "extract_keywords": keywords,
                    "filter_id": filter_id,
                    "search_range": search_time,
                }
            ]

        else:
            date, start_time, end_time = self.get_range(
                question, date, time, event_data, date_docs
            )
            docs_in_range = self.database_t.custom_query(
                query_texts=[query],
                n_results=3,
                where={
                    "$and": [
                        {"start_time": {"$gte": start_time}},
                        {"end_time": {"$lte": end_time}},
                        {"date": {"$eq": date}},
                    ]
                },
                filter_first=True,
            )
            query_result = [
                {
                    "query_range": [date, start_time, end_time],
                    "docs": docs_in_range,
                    "start_time": start_time,
                    "end_time": end_time,
                    "date": date,
                }
            ]

        return query_result, question, formatted_options

    def single_query(self, query_dict, event_data, date_event_data):
        query_result, question, formatted_options = self.query(
            query_dict, event_data, date_event_data
        )
        return query_result, question, formatted_options

    def query_all(self, query_data, event_data, date_event_data):
        all_query_results = []

        for query_dict in tqdm(query_data, desc="Processing queries"):
            try:
                query_result, question, formatted_options = self.single_query(
                    query_dict, event_data, date_event_data
                )

                all_query_results.append(
                    {
                        "metadata": query_dict,
                        "result": query_result,
                        "question": question,
                        "formatted_options": formatted_options,
                    }
                )
            except Exception as e:
                print(f"error {e}")
                continue

        return all_query_results

    def parse_query(self, data_dict):
        """
        Generate a question prompt for a model based on the given dictionary.

        Args:
            data_dict (dict): A dictionary containing date, time, query, options, and answer.

        Returns:
            str: A formatted prompt for the model.
        """
        # Extract values from the dictionary
        query_time = data_dict.get("query_time", None)

        date = query_time.get("date", "DAY6")
        time = query_time.get("time", "18000000")
        question = data_dict.get("question", "UNKNOWN_QUERY")
        query = data_dict.get("keywords", "UNKNOWN_QUERY")
        print("get keywords", query)
        # Extract choices from the dictionary
        options = []
        for letter in ["a", "b", "c", "d"]:
            choice_key = f"choice_{letter}"
            if choice_key in data_dict:
                options.append(f"{letter.upper()}. {data_dict[choice_key]}")

        # Format the options
        formatted_options = "\n".join(options)

        return question, query, formatted_options, time, date

    # CoT filter get evidence
    def extract_evidence(self, question, query_range, modality="video"):
        def parse_evidence_output(response):
            """
            Parse the LLM output and return a dictionary with the appropriate format.

            Parameters:
            - response (str): The response from the LLM.

            Returns:
            - dict: A dictionary with the extracted status and information (if applicable).
            """
            if "I can't provide evidence." in response:
                return {"status": False}

            match = re.search(r"I can provide evidence\. Evidence: (.+)", response)
            if match:
                return {"status": True, "information": match.group(1).strip()}

            return {"status": False}

        doc_ids = []
        evidence = []
        caption_results = {}
        single_range = query_range[0]["docs"]
        for index, id in enumerate(single_range["ids"]):
            video_start_time = (
                single_range["metadatas"][index]["video_path"]
                .split(".")[0]
                .split("_")[-1]
            )
            start_time = single_range["metadatas"][index]["start_time"]
            end_time = single_range["metadatas"][index]["end_time"]
            video_path = single_range["metadatas"][index]["video_path"]
            video_date = single_range["metadatas"][index]["date"]

            all_result = self.database_t.get_caption(id=id, n_result=10)

            full_caption = " ".join(all_result["documents"])
            video_caption = f"""
            Video Time: It is DAY{video_date}, {start_time} to {end_time}.
            Video Conetent: {full_caption}
            """
            caption_key = f"video{index+1}"
            caption_results[caption_key] = video_caption
            doc_ids.append(single_range["ids"][index])
            if modality == "video":
                video_prompt = f"""
                    Please analyze the provided video and help me determine if the provided information is helpful for answering the given question: {question}

                    - If the video segment is relevant, extract the relevant information from the video segment that helps answer the question. Return the response in the following JSON format:

                        \"I can provide evidence. Evidence: <state relevant information from the video that helps answer the question>\"

                    - If the video segment is not relevant, set the "status" value to "false" and return the following format:

                        \"I can't provide evidence.\"
                """

                response = self.model.inference_video(
                    video_path,
                    int(video_start_time),
                    start_time,
                    end_time,
                    video_prompt,
                )
                print(response)
                evidence.append(parse_evidence_output(response))
            elif modality == "text":
                text_prompt = f"""

                video caption: {video_caption}
                
                Format explanation:
                - The video caption follows this format:
                    "Video Time: It is DAY<video_date>, <video_start_time> to <video_end_time>. Video Content: <video_caption>"
                - Time format is HHMMSSFF (Hours, Minutes, Seconds, Frames).
                - The video content consists of captions describing events in the video from a first-person perspective.

                Please analyze the provided text, which consists of video captions describing events in the video clip from a first-person perspective.

                Help me determine if the provided information is useful for answering the given question: {question}

                If the text is relevant:
                - Extract the relevant information from the text that helps answer the question.
                - Return the response in the following format:

                    "I can provide evidence. Evidence: <state relevant information from the text>"

                If the text is not relevant:
                - Return the following format:

                    "I can't provide evidence."
                """

                response = call_gpt4(text_prompt)
                print(response)
                evidence.append(parse_evidence_output(response))
            else:
                raise ValueError(f"Invalid modality: {modality}")

        evidence_cards = []
        for index, id in enumerate(doc_ids):
            if evidence[index]["status"]:
                evidence_card = {"time": id}

                if evidence[index]["status"]:
                    evidence_card["evidence_info"] = evidence[index]["information"]
                else:
                    evidence_card["evidence_info"] = "No evidence information."

                evidence_cards.append(evidence_card)

        return evidence_cards, caption_results
