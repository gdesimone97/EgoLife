import re
import argparse
import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import random
from egorag.database.Chroma import Chroma
from egorag.agents.RagAgent import RagAgent,call_gpt4
from egorag.models import import_model


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_video_paths(base_path, name):
    video_dir = Path(base_path) / name
    return [str(video_dir / video) for video in os.listdir(video_dir)]

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

# CoT filter get evidence
def extract_evidence(question, query_range, rag_agent, modality='video'): 
    
    doc_ids = []
    evidence = []
    
    single_range = query_range[0]["docs"]
    for index, id in enumerate(single_range["ids"]):
        
        doc_ids.append(single_range["ids"][index])
        if modality == 'video':
            video_start_time = single_range["metadatas"][index]["video_path"].split('.')[0].split('_')[-1]
            start_time = single_range["metadatas"][index]["start_time"]
            end_time = single_range["metadatas"][index]["end_time"]
            video_path = single_range["metadatas"][index]["video_path"]

            video_prompt=f"""
                Please analyze the provided video and help me determine if the provided information is helpful for answering the given question: {question}

                - If the video segment is relevant, extract the relevant information from the video segment that helps answer the question. Return the response in the following JSON format:

                    \"I can provide evidence. Evidence: <state relevant information from the video that helps answer the question>\"

                - If the video segment is not relevant, set the "status" value to "false" and return the following format:

                    \"I can't provide evidence.\"
            """
    
            response = rag_agent.model.inference_video(video_path, int(video_start_time), start_time, end_time, video_prompt)
            evidence.append(parse_evidence_output(response))
        elif modality == 'text':
            text_prompt=f"""
            video caption:
            Please analyze the provided text, which consists of video captions describing events in the video clip from a first-person perspective.
            
            Help me determine if the provided information is helpful for answering the given question: {question}

            - If the text is relevant, extract the relevant information from the text that helps answer the question, and set the "status" value to "true". Return the response in the following JSON format:

                \"I can provide evidence. Evidence: <state relevant information from the text that helps answer the question>\"

            - If the text is not relevant, set the "status" value to "false" and return the following JSON format:

                \"I can't provide evidence.\"
            """

            response = call_gpt4(text_prompt)
            print(response)
            evidence.append(parse_evidence_output(response))
        else:
            raise ValueError(f"Invalid modality: {modality}")
    
    evidence_cards = []
    for index, id in enumerate(doc_ids):

        if evidence[index]["status"]:

            evidence_card = {"time" : id}

            if evidence[index]["status"]:
                evidence_card["evidence_info"] = evidence[index]["information"]
            else:
                evidence_card["evidence_info"] = "No evidence information."
 
            evidence_cards.append(evidence_card)

    return evidence_cards
def extract_single_option_answer(answer):
        """
        Extracts the single option (A, B, C, or D) from the given answer string.

        Args:
            answer (str): The answer string containing the selected option.

        Returns:
            str: The selected option (A, B, C, or D).
        """
        match = re.search(r'\b([A-D])\b', answer)
        if match:
            return match.group(1)
        return None
# use filtered and collected evidence_cards to get answer from language model
def get_answer(question, question_time,formatted_options, evidence_cards, caption_results):

    combined_string = "\n\n".join(
        f"time: {d.get('time', 'N/A')}\n"
        f"evidence_info: {d.get('evidence_info', 'No evidence info')}\n"
        for d in evidence_cards
    )

    prompt_with_evidence = f"""You will receive a series of evidence cards. Each card provides information in the following format:
    - **time**: DAYX_HHMMSSFF_HHMMSSFF  
    - **evidence_info**: Extracted visual information about the event.  

    Where **time** records a period during which specific events or actions occurred. `DAYX` indicates the day on which the events happened. `HHMMSSFF` specifies the exact time range, where `HH` is hours(0-24), `MM` is minutes)0-60), `SS` is seconds(0-60), and `FF(0-20)`.

    **Evidence cards:**  
    {combined_string}

    Using the information provided in the evidence cards, analyze the events and actions to answer the following question:

    **Question:**  
    {question}

    **Question time:**  
    {question_time}

    **Options:**  
    {formatted_options}

    ### Instructions:
    - Use the evidence cards to determine the most appropriate answer.  
    - Provide your answer as the letter corresponding to the correct option from the given choices.  
    - To calculate time differences, assume that:
        - If the question refers to **"yesterday"**, the target time is one day before the **Question time**.
        - If the question refers to **"the day before yesterday"**, the target time is two days before the **Question time**.
        - If the question refers to **"the day before the last event"**, compute the difference between the **Question time** and the time of the last event.
        - Pay attention to the **DAYX** and calculate the correct day difference.
    - Respond with **the letter only** (e.g., `A`, `B`, `C`, or `D`)."""


    prompt_without_evidence = f"""You will receive a series of **related events**, which provide contextual information relevant to the question. Each related event is structured as follows:
    - **time**: DAYX_HHMMSSFF_HHMMSSFF  
    - **event_info**: Descriptive information about the event.  

    Where **time** records a period during which specific events or actions occurred. `DAYX` indicates the day on which the events happened. `HHMMSSFF` specifies the exact time range, where `HH` is hours(0-24), `MM` is minutes)0-60), `SS` is seconds(0-60), and `FF(0-20)` is frames.

    **Related video events:**  
    {caption_results}

    Using the information provided in the related events, analyze the context and answer the following question:

    **Question:**  
    {question}

    **Question time:**  
    {question_time}

    **Options:**  
    {formatted_options}

    ### Instructions:
    - Refer to all the related events to determine the most appropriate answer.  
    - Base your reasoning on the provided information and logical inference.  
    - To calculate time differences, assume that:
        - If the question refers to **"yesterday"**, the target time is one day before the **Question time**.
        - If the question refers to **"the day before yesterday"**, the target time is two days before the **Question time**.
        - If the question refers to **"the day before the last event"**, compute the difference between the **Question time** and the time of the last event.
        - Pay attention to the **DAYX** and calculate the correct day difference.
    - Provide your answer as the letter corresponding to the correct option from the given choices.  
    - Respond with **the letter only** (e.g., `A`, `B`, `C`, or `D`)."""
    if evidence_cards==[]:
        ans=call_gpt4(prompt=prompt_without_evidence,system_message="You are an AI assistant that answers questions based on video content.",temperature=0.1)
    else:
        ans = call_gpt4(prompt=prompt_with_evidence,system_message="You are an AI assistant that answers questions based on video content.",temperature=0.1)
    if ans == "error":
        choices = ['A', 'B', 'C', 'D']
        ans="answer error, random choice"
# 随机选择一个字符
        answer_option = random.choice(choices)
    else:
      
        answer_option = extract_single_option_answer(ans)
    
    return ans, answer_option

def main(args):
    # Load configuration
    import os
    os.environ['WANDB_DISABLED']='true'
    NAME = args.name
    DATE = args.date
    DB_NAME=args.db_name
    RUN_TIME=datetime.now().strftime('%Y%m%d_%H%M%S')
    config = load_config(args.config)

    if DATE == 'all':
        video_list = []
        video_dir = os.path.join(args.video_dir, NAME)
        for day in os.listdir(video_dir):
            day_path = os.path.join(video_dir, day)
            if os.path.isdir(day_path):
                video_list.extend(
                    [os.path.join(day_path, v) for v in os.listdir(day_path)])
    else:
        video_dir = os.path.join(args.video_dir, NAME)
        day_path=os.path.join(video_dir,DATE)
        video_list = [
            os.path.join(day_path, video)
            for video in os.listdir(day_path)
        ]
    video_list.sort()

    # print(video_list[-1800:-1000])
        
    # db_name = f"{NAME}_{DATE}_{config['model']['name']}_{date}"
    # hard code database name for now
    db_name=DB_NAME
    model_class = import_model(config['model']['name'])
    with open(args.query_json, 'r', encoding='utf-8') as json_file:
        query_data = json.load(json_file)
    print(f"process {args.query_json}")
    query_json_name = os.path.splitext(os.path.basename(args.query_json))[0]
    # Initialize RAG agent
    rag_agent = RagAgent(model=model_class(**config['model']['params']),
                         database_t=Chroma(name=db_name), # preset database path
                         video_base_dir=video_dir,
                         name=NAME)

    if 'create' in args.stage:
        caption_args = config['caption']
        rag_agent.create_database_from_query(
            video_paths=video_list,
            query_json=query_data,
            caption_args=caption_args,
            human_query='Imagine you are the character in the video, describing from a first-person perspective what you saw, and everything that happened over time. Use I as the subject.',
            system_message='You are an egocentric agent. Take yourself as the main character of the video.',
            rag='rag_CLIP_t',
        )
    
    if 'query' in args.stage:
        print("query stage")
        # Generate filename with current time and db_name
        output_dir = './egolife_final_query_results'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{RUN_TIME}_{db_name}_results.json"
        output_filepath = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_filepath):

            event_file = f'events_diary_0218/{DB_NAME}_l2/merged_day_1to7.json'
            date_event_file = f'events_diary_0218/{DB_NAME}_l3/merged_day_1to7.json'
            with open(event_file, 'r') as f:
                event_data = json.load(f)
            with open(date_event_file,'r')  as  f:
                date_event_data=json.load(f)
            
            query_results = rag_agent.query_all(query_data=query_data, event_data=event_data,date_event_data=date_event_data)
            # Save query results to folder ./query_results
            os.makedirs(output_dir, exist_ok=True)
            
            # Save query results to the file
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                json.dump(query_results, outfile, ensure_ascii=False, indent=4)
            
            print(f"Query results saved to {output_filepath}")
        
        else:

            print("find cached query_results, skip query")
            with open(output_filepath, 'r', encoding='utf-8') as infile:
                query_results = json.load(infile)

    if 'answer' in args.stage:
        print("answer stage")
        output_dir = './egolife_final_final_results'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{RUN_TIME}_{db_name}_results.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Load query results either from specified file or from previous stage
        if args.query_result_json:
            with open(args.query_result_json, 'r', encoding='utf-8') as infile:
                query_results = json.load(infile)
            print(f"Loaded query results from {args.query_result_json}")
        elif 'query_results' not in locals():
            raise ValueError("No query results available. Either run 'query' stage or provide --query_result_json")
        
        answers = []
        for query_result in tqdm(query_results, desc="Answering queries"):
            try:
                query_range = query_result["result"]
                question = query_result["question"]
                question_time = query_result["metadata"]["query_time"]
                formatted_question_time=f"{question_time['date']} {question_time['time']}"
                formatted_options = query_result["formatted_options"]

                evidence_cards,caption_results = rag_agent.extract_evidence(question, query_range,modality='text')
                
                ans, answer_option = get_answer(question, formatted_question_time,formatted_options, evidence_cards,caption_results=caption_results)

                answers.append({    
                    'metadata': query_result['metadata'],
                    'model_answer': ans,
                    'model_option': answer_option,
                    'evidence_card':evidence_cards
                })
            except Exception as e:
                print(f"Error processing query: {e}")
                continue

        try:
            acc = rag_agent.calculate_accuracy(answers, save_to_file=output_filepath)
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            acc = 0
        print("Accuracy is:", acc) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process video data for RAG agent.')
    
    parser.add_argument('--date',
                        type=str,
                        required=False,
                        default="DAY1",
                        help='Date of the video directory')
    parser.add_argument('--name',
                        type=str,
                        required=False,
                        default="A1_JAKE",
                        help='Name of the video directory')
    parser.add_argument('--db_name',
                        type=str,
                        required=False,
                        default="DEFAULT_DB_NAME",
                        help='Name of the database')
    parser.add_argument('--video_dir',
                        type=str,
                        required=False,
                        default="/mnt/sfs-common/jkyang/hmguo/Egolife_all/datasets--Egolife-v1--Egolife_all/snapshots/5200df75d9aad61c35cf093ea7fb44223d3227fe/data/train",
                        help='video directory')
    parser.add_argument('--config',
                        type=str,
                        default='config/egogpt_api.yaml',
                        help='Path to config file')
    parser.add_argument('--stage',   # options: 'create', 'query', 'answer'
                        nargs='+',
                        default=['query'], 
                        required=False,
                        help='Stages to run: create database or query or answer')

    parser.add_argument('--query_json',
                        type=str,
                        default='./translated_JAKE_DAY3_test.json',
                        help='Path to the query JSON file')
    
    parser.add_argument('--query_result_json',
                        type=str,
                        default=None,
                        help='Path to existing query results JSON file')
    
    args = parser.parse_args()

    main(args)  