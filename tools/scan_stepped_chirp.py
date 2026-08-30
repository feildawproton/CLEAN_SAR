#!/usr/bin/env python3
"""
scan_stepped_chirp.py: Scans the Umbra Open Data S3 catalog to identify stepped-chirp
(multi-channel / multi-subband) collections without downloading full multi-gigabyte files.

Uses S3 HTTP Range requests (fetching only the first ~32KB XML header per CPHD).
"""

import os
import re
import sys
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import lxml.etree as etree

S3_BUCKET = "umbra-open-data-catalog"
CACHE_FILE = "/tmp/umbra_s3_catalog_cache.json"


def fetch_cphd_header_s3(s3_key):
    """
    Downloads only the first 32KB of a CPHD file from S3 using a byte-range request,
    parsing the CPHD XML metadata header.
    """
    cmd = [
        "aws", "s3api", "get-object",
        "--no-sign-request",
        "--bucket", S3_BUCKET,
        "--key", s3_key,
        "--range", "bytes=0-32768",
        "-"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        data = res.stdout
    except Exception:
        return None

    match = re.search(rb"<CPHD.*?</CPHD>", data, re.DOTALL)
    if not match:
        return None

    try:
        root = etree.fromstring(match.group(0))
    except Exception:
        return None

    num_channels_str = root.findtext("{*}Data/{*}NumChannels")
    num_channels = int(num_channels_str) if num_channels_str and num_channels_str.isdigit() else 1

    tx_seq = root.find("{*}TxSequence") is not None
    ch_params = root.findall("{*}Channel/{*}Parameters")
    channels = [c.findtext("{*}Identifier") for c in ch_params]
    fxcs = [c.findtext("{*}FxC") for c in ch_params]

    is_stepped = (num_channels > 1) or tx_seq or (len(channels) > 1)

    return {
        "key": s3_key,
        "base_name": os.path.basename(s3_key).replace("_CPHD.cphd", ""),
        "num_channels": num_channels,
        "tx_sequence": tx_seq,
        "channels": channels,
        "fxcs": fxcs,
        "is_stepped_chirp": is_stepped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scan Umbra S3 CPHD headers via byte-range requests for stepped-chirp collections."
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of CPHDs to inspect (default: 100, use 0 for all).")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent S3 range request threads (default: 16).")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index in catalog cache (default: 0).")
    args = parser.parse_args()

    if not os.path.exists(CACHE_FILE):
        print(f"Cache file {CACHE_FILE} not found. Please run list_latest_umbra.py first.")
        return

    with open(CACHE_FILE, "r") as fp:
        raw_items = json.load(fp)

    cphd_keys = [item["path"] for item in raw_items if item["path"].endswith("_CPHD.cphd")]
    total_cphds = len(cphd_keys)

    target_keys = cphd_keys[args.start_idx:] if args.num_samples == 0 else cphd_keys[args.start_idx:args.start_idx + args.num_samples]

    print(f"[*] Scanning {len(target_keys)} CPHD headers over S3 Byte-Range requests (out of {total_cphds} total in catalog)...")
    print(f"[*] Concurrency: {args.workers} workers")

    stepped_found = []
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_cphd_header_s3, k): k for k in target_keys}
        for fut in as_completed(futures):
            processed += 1
            res = fut.result()
            if res and res["is_stepped_chirp"]:
                stepped_found.append(res)
                print(f"\n[!] FOUND STEPPED CHIRP COLLECTION: {res['base_name']}")
                print(f"    Channels: {res['channels']} | FxC: {res['fxcs']} | TxSequence: {res['tx_sequence']}")
            if processed % 50 == 0 or processed == len(target_keys):
                print(f"    Progress: {processed}/{len(target_keys)} checked... ({len(stepped_found)} stepped chirp found)", end="\r")

    print("\n" + "=" * 80)
    print(f"Scan complete: Checked {processed} files. Found {len(stepped_found)} stepped chirp collections.")
    print("=" * 80)
    if stepped_found:
        for item in stepped_found:
            print(f"- {item['base_name']}: Channels={item['channels']}, FxC={item['fxcs']}")
    else:
        print("All inspected collections were single-channel spotlight chirps.")


if __name__ == "__main__":
    main()
