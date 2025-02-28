# EgoRAG

EgoRAG is a retrieval-augmented generation (RAG) system that enhances memory and query capabilities, enabling personalized and long-term comprehension.

| ![teaser.png](../assets/egorag.png) |
|:---|
| <p align="justify"><b>Figure 1. The Overview of EgoRAG Project.</b> A schematic representation of the EgoRAG system that processes user queries through a multi-stage pipeline: First, it identifies relevant time ranges using multi-level summarization. Then, it retrieves pertinent video segments based on those time ranges. Finally, it gathers multiple pieces of evidence which are fed into an answering model (e.g., EgoGPT) to generate comprehensive responses to user queries.
</p>

## Installation

To install EgoRAG, make sure you are in the `EgoRAG` directory and run the following commands:

```bash
pip install -e .
```

This will install the package in editable mode, allowing you to make changes to the code without needing to reinstall it.

## Usage

### Question Preparation

To use the EgoRAG system, questions must be in the following format:

```json
{
    "ID": "1",
    "query_time": {
        "date": "DAY1",
        "time": "11210217"
    },
    "type": "EntityLog",
    "need_audio": false,
    "need_name": true,
    "last_time": false,
    "trigger": "The table was filled with various tools and parts",
    "question": "Who used the screwdriver first?",
    "choice_a": "Tasha",
    "choice_b": "Alice",
    "choice_c": "Shure",
    "choice_d": "Lucia",
    "answer": "B",
    "target_time": {
        "date": "DAY1",
        "time": "11152408"
    },
    "keywords": "use screwdriver",
    "reason": "Saw Alice tightening screws with a screwdriver",
}
```

The required fields are `query_time`, `question`, `choice`, and `keywords`.

### Creating the Database

There are two methods to create a database for querying and answering questions:

#### Method 1: Create from JSON (Fast)

If you have captions for each video, you can gather them into a single JSON file and generate the database. The JSON should follow this format:

```json
{
    "start_time": "11100000",
    "end_time": "11103000",
    "text": "<model caption>",
    "date": "DAY1",
    "video_path": "Egolife/train/A1_JAKE/DAY1_A1_JAKE_11100000.mp4"
}
```

To create the database, run:

```bash
python3 create_database.py --db_name "DB_NAME" --json_path "/path/to/your/database.json"
```

This will generate the database `<DB_NAME>`.

#### Method 2: Create from Video (Slow)

In this method, the model will go through all the videos, extract captions, and add them to the database one by one. Use the following command to create the database:

```bash
python3 main.py \
    --name "NAME"\
    --db_name "DB_NAME"\     
    --stage create  \   
    --video_dir "video_base_dir" \    
    --config "config/your_model_config.yaml"
```

### Querying the Database

To improve query performance, an event summary must be generated for each database. Use the following command to generate the event summary:

```bash
python3 gen_event.py --db_name <DB_NAME> --diary_dir "folder/to/save/diary"
```

Then, run:

```bash
python3 main.py  \   
    --name "NAME" \    
    --db_name "DB_NAME" \    
    --stage create  \   
    --video_dir "video_base_dir"\     
    --config "config/your_model_config.yaml"\     
    --query_json "path/to/your/query.json" \    
    --diary_dir "folder/to/save/diary"\     
    --query_result "folder/to/save/result"\
```

### Answering Queries

To answer specific query results, use the following command:

```bash
python3 main.py\     
    --name "NAME"\     
    --db_name "DB_NAME"\     
    --stage answer \    
    --video_dir "video_base_dir"\     
    --config "config/your_model_config.yaml" \    
    --query_json "path/to/your/query.json"\     
    --query_result_json "path/to/query/result.json"\     
    --answer_result "folder/to/answer/result"\
```

Alternatively, you can automatically answer after querying with the following command:

```bash
python3 main.py \    
    --name "NAME"\     
    --db_name "DB_NAME"\     
    --stage query_answer\     
    --video_dir "video_base_dir" \    
    --config "config/your_model_config.yaml"\     
    --query_json "path/to/your/query.json"\     
    --query_result "folder/to/save/result"\     
    --answer_result "folder/to/answer/result"\
```

