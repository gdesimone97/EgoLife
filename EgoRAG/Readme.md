
# EgoRAG

EgoRAG is a retrieval-augmented generation (RAG) system that enhances memory and query capabilities, enabling personalized and long-term comprehension.

## Installation

To install EgoRAG, run the following commands:

```
git clone <egorag-url>
cd <egorag-url>
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
    "type_chinese": "实体日志",
    "need_audio": false,
    "need_name": true,
    "last_time": false,
    "trigger": "The table was filled with various tools and parts",
    "trigger_chinese": "桌上摆满了各种工具和零件",
    "question": "Who used the screwdriver first?",
    "question_chinese": "谁最先使用过螺丝刀？",
    "choice_a": "Tasha",
    "choice_a_chinese": "小薇",
    "choice_b": "Alice",
    "choice_b_chinese": "爱丽丝",
    "choice_c": "Shure",
    "choice_c_chinese": "修硕",
    "choice_d": "Lucia",
    "choice_d_chinese": "露西",
    "answer": "B",
    "target_time": {
        "date": "DAY1",
        "time": "11152408"
    },
    "keywords": "use screwdriver",
    "reason": "Saw Alice tightening screws with a screwdriver",
    "reason_chinese": "看见爱丽丝用螺丝刀紧螺丝"
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

```
python3 create_database.py --db_name "DB_NAME" --json_path "/path/to/your/database.json"
```

This will generate the database `<DB_NAME>`.

#### Method 2: Create from Video (Slow)

In this method, the model will go through all the videos, extract captions, and add them to the database one by one. Use the following command to create the database:

```
python3 main.py \
    --name "NAME"\
    --db_name "DB_NAME"\     
    --stage create  \   
    --video_dir "video_base_dir" \    
    --config "config/your_model_config.yaml"
```

### Querying the Database

To improve query performance, an event summary must be generated for each database. Use the following command to generate the event summary:

```
python3 gen_event.py --db_name <DB_NAME> --diary_dir "folder/to/save/diary"
```

Then, run:

```
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

```
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

```
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

