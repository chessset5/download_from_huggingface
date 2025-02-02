import huggingface_hub
import os

CACHE_DIR = r"D:\TEMP\hugging_face_cache"

BASE_DIR = r"E:\llms"


def download_from_huggingface(repo_id: str, local_dir: str) -> None:
    """
    creates a new directory and downloads a snapshot from hugging face into said directory
    """
    # repo_id = "username/repo_name"  # Replace with the actual repository ID
    # local_dir = "./local_directory"  # Replace with your desired local directory+
    normalized_dir = os.path.normpath(local_dir)
    os.makedirs(normalized_dir, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=repo_id, 
        local_dir=normalized_dir, 
        cache_dir=CACHE_DIR,
        max_workers=1,
    )


def make_dir_paths(id_list: list[str], base_dir: str) -> list[tuple[str, str]]:
    """
    makes new directories base off the base directory and the id_list
    returns repos [(id,new_dir)]
    """
    repos = list[tuple[str, str]]()
    for current_id in id_list:
        if current_id == r"":
            continue
        new_dir: str = os.path.join(base_dir, current_id)
        new_dir: str = os.path.normpath(path=new_dir)
        repos.append((current_id, new_dir))

    return repos


def main() -> None:
    """
    main function
    it will init the repos to download and loop through them and send them to
    download_deepseek
    """

    base_dir = BASE_DIR

    id_list: list[str] = [
        r"deepseek-ai/DeepSeek-V3-Base",
        r"deepseek-ai/DeepSeek-V3",
        r"deepseek-ai/DeepSeek-R1",
        r"deepseek-ai/DeepSeek-R1-Zero",
        r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        r"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        r"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        r"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        r"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        r"",
    ]

    repos: list[str] = make_dir_paths(id_list=id_list, base_dir=base_dir)

    for repo in repos:
        print(f"Current REPO ID:{repo[0]}")
        download_from_huggingface(repo_id=repo[0], local_dir=repo[1])


if "__main__" == __name__:
    main()
