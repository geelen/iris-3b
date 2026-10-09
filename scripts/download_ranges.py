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
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-mib", type=int, default=32)
    parser.add_argument("--sparse-source", type=Path)
    args = parser.parse_args()
    meta = get_hf_file_metadata(hf_hub_url(args.repo, args.file))
    size = meta.size
    if args.destination.exists() and args.destination.stat().st_size == size:
        print(f"Already downloaded: {args.destination}", flush=True)
        return
    chunks = args.destination.parent / ".ranges" / meta.etag
    chunks.mkdir(parents=True, exist_ok=True)
    block = args.block_mib * 1024 * 1024
    count = (size + block - 1) // block
    if args.sparse_source:
        recovered = 0
        with args.sparse_source.open("rb") as source:
            for index in range(count):
                start = index * block
                length = min(block, size - start)
                source.seek(start)
                data = source.read(length)
                # A missing/incomplete sparse transfer chunk contains a zero tail.
                # Reused chunks are provisional until the final SHA256 matches.
                if len(data) == length and any(data[-4096:]) and any(data[:4096]):
                    (chunks / f"{index:05d}").write_bytes(data)
                    recovered += 1
        print(f"Recovered {recovered}/{count} provisional ranges; full SHA256 check required", flush=True)

    def download(index):
        start = index * block
        end = min(size, start + block) - 1
        path = chunks / f"{index:05d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        temporary = path.with_suffix(".partial")
        for attempt in range(20):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset > end - start + 1:
                    raise RuntimeError("Partial range exceeds expected length")
                if offset == end - start + 1:
                    temporary.replace(path)
                    return
                resume = start + offset
                with requests.get(meta.location, headers={"Range": f"bytes={resume}-{end}"}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f"bytes {resume}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Incorrect range response: {response.status_code} {response.headers.get('Content-Range')}")
                    with temporary.open("ab") as file:
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
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
