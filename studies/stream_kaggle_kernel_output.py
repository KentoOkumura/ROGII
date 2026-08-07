"""Stream one Kaggle kernel output file without buffering it in RAM.

Kaggle CLI 2.2.3 calls ``requests.get(..., stream=True)`` but then writes
``response.content``.  That can exhaust memory for a large single artifact.
This utility uses the same authenticated output listing and signed URL, but
writes fixed-size chunks while calculating SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("remote_file", help="exact output file name")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=8)
    return parser.parse_args()


def find_output_url(api: KaggleApi, kernel: str, remote_file: str) -> str:
    owner, slug, _ = api.parse_kernel_string(kernel)
    token: str | None = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 200
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(
                request
            )
            for item in response.files or []:
                if item.file_name == remote_file:
                    return str(item.url)
            token = response.next_page_token
            if not token:
                break
    raise FileNotFoundError(f"{remote_file!r} is not an output of {kernel!r}")


def main() -> None:
    args = parse_args()
    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    args.destination.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    url = find_output_url(api, args.kernel, args.remote_file)
    digest = hashlib.sha256()
    total = 0
    chunk_bytes = args.chunk_mib * 1024 * 1024
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length", "0"))
        with args.destination.open("xb") as output:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if not chunk:
                    continue
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total % (256 * 1024 * 1024) < chunk_bytes:
                    print(f"downloaded_bytes={total}", flush=True)
    if expected and total != expected:
        raise RuntimeError(f"downloaded {total} bytes, expected {expected}")
    print(f"bytes={total}")
    print(f"sha256={digest.hexdigest()}")
    print(f"path={args.destination}")


if __name__ == "__main__":
    main()
