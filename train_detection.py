import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(script_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_main():
    script_path = Path(__file__).resolve().parent / "scripts" / "train_detection.py"
    module = _load_module(script_path, "project_train_detection")
    return module.main


def _ensure_default_hierarchical_dataset() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return
    if any(arg.startswith("--data") for arg in sys.argv[1:]):
        return

    project_root = Path(__file__).resolve().parent
    hierarchical_yaml = project_root / "data" / "detection_hierarchical" / "hierarchical_detection.yaml"
    if hierarchical_yaml.exists():
        return

    source_yaml = project_root / "data" / "detection_merged_umfih" / "merged_detection.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(
            "Default hierarchical dataset is missing and source merged dataset was not found at "
            f"{source_yaml}"
        )

    out_root = hierarchical_yaml.parent
    prep_script = project_root / "scripts" / "prepare_hierarchical_detection_dataset.py"
    prep_module = _load_module(prep_script, "project_prepare_hierarchical_detection_dataset")
    print(f"Preparing hierarchical dataset from {source_yaml} -> {out_root}")
    prep_module.prepare_hierarchical_dataset(data_path=source_yaml, out_root=out_root)


if __name__ == "__main__":
    _ensure_default_hierarchical_dataset()
    _load_main()()
