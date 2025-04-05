import os
from huggingface_hub import snapshot_download

os.chdir("/home/dsmgpp000/datasets/Ego")

local_path = snapshot_download(
    repo_id="EgoGPT/EgoIT_Video", 
    repo_type="dataset", 
    local_dir="data"
)