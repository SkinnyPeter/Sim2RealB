from pathlib import Path
from huggingface_hub import login, HfApi

login()

api = HfApi()
repo_id = "Sim2RealB/Sim2RealB_dataset"
commit_message="Uploaded 20250804_105355.h5 WE HAVE BOTH TRAJ+H5 WITH 20250804_105355"

who = api.whoami()
print("Logged in as:", who["name"])

info = api.repo_info(repo_id=repo_id, repo_type="dataset")
print("Repo found:", info.id)

# Go to project root, then /data
data_path = Path(__file__).resolve().parents[2] / "data"

api.upload_folder(
    folder_path=str(data_path),
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=commit_message,
    ignore_patterns=[
        "**/.git/**",
        "**/__pycache__/**",
        "*.py",
        "*.log",
    ],
)

print("Upload complete")