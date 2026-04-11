import re
import subprocess
import sys
from typing import Optional


def run(cmd):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def get_nvidia_smi() -> str:
    res = run(["nvidia-smi"])
    if res.returncode != 0:
        return ""
    return res.stdout


def parse_cuda_version(nvidia_smi_output: str) -> Optional[str]:
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", nvidia_smi_output)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def choose_torch_index(cuda_version: Optional[str]) -> str:
    # Stability-first fallback strategy:
    # - older drivers (like CUDA 11.0-era) usually work best with cu118 wheels
    # - if parsing fails, install CPU wheel to avoid runtime mismatches
    if cuda_version is None:
        return "cpu"
    major_minor = tuple(map(int, cuda_version.split(".")))
    if major_minor >= (11, 0):
        return "cu118"
    return "cpu"


def install_torch(channel: str) -> int:
    base = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.run(base, check=False)

    if channel == "cu118":
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    return subprocess.run(cmd, check=False).returncode


def main():
    smi = get_nvidia_smi()
    if smi:
        print(smi)
    cuda_version = parse_cuda_version(smi)
    channel = choose_torch_index(cuda_version)
    print(f"Detected CUDA from driver: {cuda_version}, selected PyTorch channel: {channel}")
    code = install_torch(channel)
    if code != 0:
        print("Torch installation failed.")
        sys.exit(code)
    print("Torch installation done.")
    print("Run: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")


if __name__ == "__main__":
    main()
