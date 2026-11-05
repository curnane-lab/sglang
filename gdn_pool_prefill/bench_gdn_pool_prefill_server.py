import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def tokens(seed, length):
    return [1000 + ((seed + index * 17) % 90000) for index in range(length)]


def post(url, endpoint, payload=None):
    data = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        url + endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def invoke(url, rid, input_ids, output_len):
    payload = json.dumps(
        {
            "rid": rid,
            "input_ids": input_ids,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        url + "/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter_ns()
    first_token = None
    output_ids = []
    text_parts = []
    meta = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            body = raw_line[6:].strip()
            if body == b"[DONE]":
                continue
            item = json.loads(body)
            if "error" in item:
                raise RuntimeError(item["error"])
            if first_token is None and (item.get("text") or item.get("output_ids")):
                first_token = time.perf_counter_ns()
            text_parts.append(item.get("text") or "")
            ids = item.get("output_ids") or []
            item_meta = item.get("meta_info") or {}
            if ids and item_meta.get("completion_tokens") == len(ids):
                output_ids = ids
            else:
                output_ids.extend(ids)
            if item_meta:
                meta = item_meta
    ended = time.perf_counter_ns()
    assert meta is not None, rid
    assert meta["finish_reason"]["type"] == "length", (rid, meta)
    assert meta["completion_tokens"] == output_len, (rid, meta)
    if output_ids:
        assert len(output_ids) == output_len, (rid, len(output_ids), output_len)
    return {
        "rid": rid,
        "input_tokens": len(input_ids),
        "cached_tokens": meta.get("cached_tokens") or 0,
        "cached_tokens_details": meta.get("cached_tokens_details"),
        "decode_tokens": meta["completion_tokens"],
        "ttft_ms": None if first_token is None else (first_token - started) / 1e6,
        "e2e_ms": (ended - started) / 1e6,
        "output_ids": output_ids,
        "text_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest(),
    }


def run_batch(url, batch_size, mode, round_index, prefix, output_len, expected_cached):
    seed = {8: 600000, 16: 800000, 32: 900000, 64: 1000000}[
        batch_size
    ] + round_index * 1000
    requests = [prefix + tokens(seed + index * 37, 128) for index in range(batch_size)]
    started = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                invoke,
                url,
                f"{mode}-b{batch_size}-r{round_index}-{index}",
                ids,
                output_len,
            )
            for index, ids in enumerate(requests)
        ]
        records = [future.result() for future in futures]
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    assert all(
        record["cached_tokens"] == expected_cached for record in records
    ), records
    return {
        "batch_size": batch_size,
        "mode": mode,
        "round_index": round_index,
        "wall_ms": wall_ms,
        "output_tokens_per_second": batch_size * output_len * 1000 / wall_ms,
        "requests": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31234")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    post(args.url, "/flush_cache")
    prefix = tokens(20000, 384)
    invoke(args.url, "legacy-warm-cold", prefix + tokens(120000, 128), args.output_len)
    legacy_warm = invoke(
        args.url, "legacy-warm-hot", prefix + tokens(130000, 128), args.output_len
    )
    pool_warm = invoke(
        args.url, "pool-warm-hot", prefix + tokens(131000, 128), args.output_len
    )
    expected_cached = max(legacy_warm["cached_tokens"], pool_warm["cached_tokens"])
    assert expected_cached > 0
    legacy_verify = invoke(
        args.url, "legacy-warm-verify", prefix + tokens(132000, 128), args.output_len
    )
    pool_verify = invoke(
        args.url, "pool-warm-verify", prefix + tokens(133000, 128), args.output_len
    )
    assert legacy_verify["cached_tokens"] == expected_cached, legacy_verify
    assert pool_verify["cached_tokens"] == expected_cached, pool_verify
    print(f"verified cached tokens on both paths: {expected_cached}", flush=True)

    rounds = []
    for round_index, mode in enumerate(
        ["legacy", "pool", "pool", "legacy"] * args.cycles
    ):
        result = run_batch(
            args.url,
            args.batch_size,
            mode,
            round_index,
            prefix,
            args.output_len,
            expected_cached,
        )
        rounds.append(result)
        print(
            f"chunk={args.chunk_size} {mode} round {round_index}: "
            f"{result['wall_ms']:.3f} ms",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "chunk_size": args.chunk_size,
                "expected_cached_tokens": expected_cached,
                "warm_cached_tokens": {
                    "legacy": legacy_warm["cached_tokens"],
                    "pool": pool_warm["cached_tokens"],
                    "legacy_verify": legacy_verify["cached_tokens"],
                    "pool_verify": pool_verify["cached_tokens"],
                },
                "rounds": rounds,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
