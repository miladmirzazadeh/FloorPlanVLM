"""Hugging Face Hub helpers: login, repo creation, resume, and 'finished' markers.

The Hub is our durable autosave target. Trainer pushes full checkpoints there during
training (hub_strategy='all_checkpoints'); these helpers let a brand-new pod pull the
latest checkpoint back and continue, and let the watchdog know a stage is already done
even if the local volume was wiped.
"""
import os

from huggingface_hub import HfApi, login, snapshot_download, create_repo

from . import config

_api = None
FINISHED = "FINISHED"


def api():
    global _api
    if _api is None:
        _api = HfApi(token=config.HF_TOKEN or None)
    return _api


def hf_login():
    if not config.HF_TOKEN:
        print("[hub] WARNING: HF_TOKEN not set — pushes/pulls will fail.")
        return
    if "/" not in config.REPO_SFT:
        print("[hub] WARNING: repo id has no owner ('{}') — set HF_USER or HF_REPO_*."
              .format(config.REPO_SFT))
    login(token=config.HF_TOKEN, add_to_git_credential=True)


def ensure_repo(repo_id):
    try:
        create_repo(repo_id, private=config.PRIVATE_REPOS, exist_ok=True,
                    token=config.HF_TOKEN or None)
    except Exception as e:
        print(f"[hub] ensure_repo({repo_id}) failed: {e}")


def is_finished(repo_id):
    try:
        return api().file_exists(repo_id=repo_id, filename=FINISHED)
    except Exception:
        try:
            return FINISHED in api().list_repo_files(repo_id=repo_id)
        except Exception:
            return False


def mark_finished(repo_id):
    try:
        ensure_repo(repo_id)
        api().upload_file(path_or_fileobj=b"done\n", path_in_repo=FINISHED, repo_id=repo_id)
        print(f"[hub] marked {repo_id} FINISHED")
    except Exception as e:
        print(f"[hub] mark_finished failed: {e}")


def pull_latest_checkpoint(repo_id, output_dir):
    """Download the highest-step checkpoint-* folder from the Hub. Returns path or None."""
    try:
        files = api().list_repo_files(repo_id=repo_id)
    except Exception:
        return None
    steps = set()
    for f in files:
        if f.startswith("checkpoint-"):
            try:
                steps.add(int(f.split("/")[0].split("-")[1]))
            except Exception:
                pass
    if not steps:
        return None
    ckpt = f"checkpoint-{max(steps)}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[hub] resuming: downloading {repo_id}/{ckpt} ...")
    try:
        snapshot_download(repo_id=repo_id, allow_patterns=f"{ckpt}/*",
                          local_dir=output_dir, token=config.HF_TOKEN or None)
    except Exception as e:
        print(f"[hub] checkpoint download failed: {e}")
        return None
    local = os.path.join(output_dir, ckpt)
    return local if os.path.isdir(local) else None


def upload_folder(repo_id, folder):
    ensure_repo(repo_id)
    api().upload_folder(folder_path=folder, repo_id=repo_id)
