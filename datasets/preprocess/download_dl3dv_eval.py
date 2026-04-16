"""
A download script to download the DL3DV/DL3DV-Evaluation dataset from the Hugging Face dataset repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tarfile
import os
from tqdm import tqdm  # type: ignore


def _require_hf_hub():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        return snapshot_download
    except Exception as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install -U huggingface_hub"
        ) from e

def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
    except Exception:
        return False
    return str(target).startswith(str(directory) + os.sep) or target == directory

def _extract_one(tar_path_str: str, out_root_str: str) -> str:
    tar_path = Path(tar_path_str)
    out_root = Path(out_root_str)
    scene_name = tar_path.stem
    scene_dir = out_root / scene_name
    if scene_dir.exists():
        return f"{scene_name}: skipped"
    scene_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf.getmembers():
            member_path = scene_dir / member.name
            if not _is_within_directory(scene_dir, member_path):
                raise RuntimeError(f"Unsafe path in tar: {member.name}")
        tf.extractall(path=scene_dir)
    return f"{scene_name}: ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download content from a Hugging Face dataset repo using snapshot_download.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("./DL3DV-Evaluation"),
        help="Destination directory where files will be materialized",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel extraction workers for untar",
    )
    args = parser.parse_args()

    snapshot_download = _require_hf_hub()

    args.dst.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading repo=DL3DV/DL3DV-Evaluation to {args.dst} "
    )

    local_path = snapshot_download(
        repo_id="DL3DV/DL3DV-Evaluation",
        repo_type="dataset",
        revision="main",
        local_dir=str(args.dst),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Download complete. Files available under: {local_path}. Untarring...")

    # Untar all scene archives from images_tar → images/<hash>
    images_tar_dir = args.dst / "images_tar"
    images_out_dir = args.dst / "images"
    if images_tar_dir.exists() and images_tar_dir.is_dir():
        images_out_dir.mkdir(parents=True, exist_ok=True)
        tars = sorted(list(images_tar_dir.glob("*.tar")))
        if not tars:
            print(f"No .tar files found in {images_tar_dir}, skipping extraction.")
        else:
            print(f"Extracting {len(tars)} archive(s) to {images_out_dir} with {args.workers} worker(s) ...")

            completed_ok = 0
            skipped = 0

            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(_extract_one, str(tp), str(images_out_dir)) for tp in tars]
                pbar = tqdm(total=len(futures), desc="Untarring", unit="tar") if tqdm else None
                try:
                    for fut in as_completed(futures):
                        try:
                            msg = fut.result()
                            if msg.endswith(": ok"):
                                completed_ok += 1
                            elif msg.endswith(": skipped"):
                                skipped += 1
                        except Exception as e:
                            print(f"Extraction error: {e}")
                        finally:
                            if pbar is not None:
                                pbar.update(1)
                finally:
                    if pbar is not None:
                        pbar.close()
            print(f"Extraction complete. Created {completed_ok}, skipped {skipped}.")
    else:
        print(f"{images_tar_dir} not found, skipping extraction.")


if __name__ == "__main__":
    main()