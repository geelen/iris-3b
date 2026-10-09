"""Download a large HF file in persistent, verified HTTP ranges."""

import argparse
import concurrent.futures
import hashlib
import shutil
import time
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("file")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    meta = get_hf_file_metadata(hf_hub_url(args.repo, args.file))
    size = meta.size
    if args.destination.exists() and args.destination.stat().st_size == size:
        print(f"Already downloaded: {args.destination}", flush=True)
        return
    chunks = args.destination.parent / ".ranges" / meta.etag
    chunks.mkdir(parents=True, exist_ok=True)
    block = 32 * 1024 * 1024
    count = (size + block - 1) // block

    def download(index):
        start = index * block
        end = min(size, start + block) - 1
        path = chunks / f"{index:05d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(20):
            try:
                with requests.get(meta.location, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Incorrect range response: {response.status_code} {response.headers.get('Content-Range')}")
                    temporary = path.with_suffix(".partial")
                    with temporary.open("wb") as file:
                        for data in response.iter_content(1024 * 1024):
                            file.write(data)
                    if temporary.stat().st_size != end - start + 1:
                        raise RuntimeError("Incomplete range")
                    temporary.replace(path)
                    return
            except Exception as exc:
                print(f"Range {index} retry {attempt + 1}: {exc}", flush=True)
                time.sleep(min(10, attempt + 1))
        raise RuntimeError(f"Range {index} exhausted retries")

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, index) for index in range(count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == count:
                print(f"{args.file}: {completed}/{count} ranges complete, {time.perf_counter() - started:.0f}s", flush=True)
    digest = hashlib.sha256()
    temporary = args.destination.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for index in range(count):
            with (chunks / f"{index:05d}").open("rb") as file:
                while data := file.read(8 * 1024 * 1024):
                    digest.update(data)
                    output.write(data)
    if len(meta.etag) == 64 and digest.hexdigest() != meta.etag:
        raise RuntimeError("SHA256 mismatch")
    temporary.replace(args.destination)
    print(f"Verified: {args.destination}, {size} bytes, SHA256 {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
