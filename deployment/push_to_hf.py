import os
from huggingface_hub import HfApi, login

HF_USERNAME = "amitcoolll"
SPACE_NAME = "predictive-maintenance-app"  # you can rename this
SPACE_REPO = f"{HF_USERNAME}/{SPACE_NAME}"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face token: ").strip()

    login(token)
    api = HfApi()

    # Create/update Docker Space
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    # Upload deployment folder (app.py, Dockerfile, requirements.txt)
    api.upload_folder(
        folder_path="deployment",
        repo_id=SPACE_REPO,
        repo_type="space",
    )

    print(f"🚀 Deployed to: https://huggingface.co/spaces/{SPACE_REPO}")

if __name__ == "__main__":
    main()
