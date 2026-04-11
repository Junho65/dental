from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files


def main():
    target = Path("data/raw/dentex")
    target.mkdir(parents=True, exist_ok=True)
    repo_id = "LUNA0206/DENTEX"
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"Found {len(files)} files in dataset repo")

    # Stability-first local workflow:
    # Download smaller/essential files first to avoid long stalls.
    preferred = [
        "README.md",
        "DENTEX/validation_triple.json",
        "DENTEX/validation_data.zip",
        "DENTEX/test_data.zip",
    ]
    selected = [f for f in preferred if f in files]
    if not selected:
        selected = files[:5]

    for f in selected:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded: {f} -> {local}")

    print("License: CC-BY-NC-SA-4.0 (check dataset card for details)")


if __name__ == "__main__":
    main()
