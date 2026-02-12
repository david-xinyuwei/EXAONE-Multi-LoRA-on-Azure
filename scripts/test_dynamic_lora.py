#!/usr/bin/env python3
"""
Test dynamic LoRA adapter loading/unloading on a running vLLM server.
Validates runtime adapter hot-swap without server restart.

Requires: VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

Author: Hyeonsang Jeon
"""
import json
import time
import requests
import sys
import argparse

VLLM_URL = "http://localhost:8080"
BASE_MODEL = "/data/EXAONE-3.5-2.4B-Instruct"

# Default adapter configs (same domains as static serving)
DEFAULT_ADAPTERS = {
    "medical": "/data/lora-adapters/medical",
    "legal": "/data/lora-adapters/legal",
    "customer_support": "/data/lora-adapters/customer_support",
    "code": "/data/lora-adapters/code",
}

TEST_PROMPTS = {
    "medical": "What are the common symptoms and treatment options for Type 2 diabetes?",
    "legal": "What are the key differences between civil law and criminal law in South Korea?",
    "customer_support": "I ordered a product 2 weeks ago and it still hasn't arrived. What should I do?",
    "code": "Write a Python function that implements binary search on a sorted list.",
}


def list_models() -> list:
    """List currently loaded models via /v1/models."""
    resp = requests.get(f"{VLLM_URL}/v1/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def load_adapter(name: str, path: str) -> dict:
    """Load a LoRA adapter at runtime."""
    payload = {"lora_name": name, "lora_path": path}
    start = time.time()
    resp = requests.post(f"{VLLM_URL}/v1/load_lora_adapter", json=payload, timeout=60)
    elapsed = time.time() - start
    resp.raise_for_status()
    return {"adapter": name, "action": "load", "latency_ms": round(elapsed * 1000, 1)}


def unload_adapter(name: str) -> dict:
    """Unload a LoRA adapter at runtime."""
    payload = {"lora_name": name}
    start = time.time()
    resp = requests.post(f"{VLLM_URL}/v1/unload_lora_adapter", json=payload, timeout=60)
    elapsed = time.time() - start
    resp.raise_for_status()
    return {"adapter": name, "action": "unload", "latency_ms": round(elapsed * 1000, 1)}


def query_model(model: str, prompt: str, max_tokens: int = 128) -> dict:
    """Send a chat completion request to vLLM."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    start = time.time()
    resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=60)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "model": model,
        "content": content,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_s": round(elapsed, 3),
    }


def test_dynamic_lifecycle():
    """Test full adapter lifecycle: load → query → unload → verify."""
    results = []
    print("=" * 80)
    print("Dynamic LoRA Adapter Lifecycle Test")
    print("=" * 80)

    # Step 1: Check initial state
    print("\n[Step 1] Initial model list")
    models = list_models()
    print(f"  Loaded models: {models}")

    # Step 2: Load adapters one by one, measure load latency
    print("\n[Step 2] Dynamic adapter loading")
    print("-" * 60)
    load_results = []
    for name, path in DEFAULT_ADAPTERS.items():
        try:
            result = load_adapter(name, path)
            load_results.append(result)
            print(f"  ✓ Loaded '{name}' in {result['latency_ms']}ms")
        except Exception as e:
            print(f"  ✗ Failed to load '{name}': {e}")
            load_results.append({"adapter": name, "action": "load", "error": str(e)})

    # Step 3: Verify all adapters are available
    print("\n[Step 3] Verify loaded adapters")
    models = list_models()
    print(f"  Loaded models: {models}")
    for name in DEFAULT_ADAPTERS:
        status = "✓" if name in models else "✗ MISSING"
        print(f"  {status} {name}")

    # Step 4: Query each adapter
    print("\n[Step 4] Query each adapter")
    print("-" * 60)
    query_results = []
    for name in DEFAULT_ADAPTERS:
        prompt = TEST_PROMPTS.get(name, "Hello, how are you?")
        try:
            result = query_model(name, prompt)
            query_results.append(result)
            print(f"  {name}: {result['completion_tokens']} tokens, "
                  f"{result['latency_s']}s")
            print(f"    → {result['content'][:150]}...")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Step 5: Unload one adapter, verify it's gone, query should fail
    print("\n[Step 5] Unload 'medical' adapter")
    print("-" * 60)
    unload_results = []
    try:
        result = unload_adapter("medical")
        unload_results.append(result)
        print(f"  ✓ Unloaded 'medical' in {result['latency_ms']}ms")
    except Exception as e:
        print(f"  ✗ Failed to unload: {e}")

    models = list_models()
    medical_gone = "medical" not in models
    print(f"  Medical removed: {'✓ Yes' if medical_gone else '✗ No'}")
    print(f"  Remaining models: {models}")

    # Step 6: Verify query to unloaded adapter fails gracefully
    print("\n[Step 6] Query unloaded adapter (should fail)")
    try:
        result = query_model("medical", TEST_PROMPTS["medical"])
        print(f"  ✗ Unexpected success - adapter still responding")
    except requests.exceptions.HTTPError as e:
        print(f"  ✓ Correctly rejected: {e.response.status_code}")
    except Exception as e:
        print(f"  ✓ Correctly rejected: {e}")

    # Step 7: Reload adapter (simulate version update)
    print("\n[Step 7] Reload 'medical' adapter (simulating update)")
    try:
        result = load_adapter("medical", DEFAULT_ADAPTERS["medical"])
        load_results.append(result)
        print(f"  ✓ Reloaded 'medical' in {result['latency_ms']}ms")
        # Verify it works again
        query = query_model("medical", TEST_PROMPTS["medical"])
        print(f"  ✓ Query after reload: {query['completion_tokens']} tokens, "
              f"{query['latency_s']}s")
    except Exception as e:
        print(f"  ✗ Reload failed: {e}")

    # Step 8: Unload all adapters
    print("\n[Step 8] Cleanup - unload all adapters")
    for name in DEFAULT_ADAPTERS:
        try:
            result = unload_adapter(name)
            print(f"  ✓ Unloaded '{name}' in {result['latency_ms']}ms")
        except Exception:
            pass

    models = list_models()
    print(f"  Final models: {models}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Action':<12} {'Adapter':<20} {'Latency':>10}")
    print("-" * 45)
    for r in load_results:
        if "error" not in r:
            print(f"{'load':<12} {r['adapter']:<20} {r['latency_ms']:>8.1f}ms")
    for r in unload_results:
        if "error" not in r:
            print(f"{'unload':<12} {r['adapter']:<20} {r['latency_ms']:>8.1f}ms")

    # Save results
    output = {
        "load_results": load_results,
        "query_results": [{k: v for k, v in r.items() if k != "content"}
                          for r in query_results],
        "unload_results": unload_results,
    }
    with open("/tmp/dynamic_lora_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to /tmp/dynamic_lora_test_results.json")


def test_swap_latency(num_cycles: int = 5):
    """Benchmark adapter swap latency over multiple cycles."""
    print("\n" + "=" * 80)
    print(f"Adapter Swap Latency Benchmark ({num_cycles} cycles)")
    print("=" * 80)

    swap_times = []
    adapter_name = "medical"
    adapter_path = DEFAULT_ADAPTERS[adapter_name]
    prompt = TEST_PROMPTS[adapter_name]

    for i in range(num_cycles):
        cycle_start = time.time()

        # Load
        load_result = load_adapter(adapter_name, adapter_path)

        # Query (verify it works)
        query_result = query_model(adapter_name, prompt, max_tokens=32)

        # Unload
        unload_result = unload_adapter(adapter_name)

        cycle_time = time.time() - cycle_start

        swap_times.append({
            "cycle": i + 1,
            "load_ms": load_result["latency_ms"],
            "query_s": query_result["latency_s"],
            "unload_ms": unload_result["latency_ms"],
            "total_s": round(cycle_time, 3),
        })
        sys.stdout.write(".")
        sys.stdout.flush()

    print(" done\n")

    # Summary
    load_times = [s["load_ms"] for s in swap_times]
    unload_times = [s["unload_ms"] for s in swap_times]

    print(f"{'Metric':<20} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("-" * 52)
    print(f"{'Load (ms)':<20} {sum(load_times)/len(load_times):>10.1f} "
          f"{min(load_times):>10.1f} {max(load_times):>10.1f}")
    print(f"{'Unload (ms)':<20} {sum(unload_times)/len(unload_times):>10.1f} "
          f"{min(unload_times):>10.1f} {max(unload_times):>10.1f}")

    with open("/tmp/swap_latency_results.json", "w") as f:
        json.dump(swap_times, f, indent=2)
    print(f"\nResults saved to /tmp/swap_latency_results.json")


def main():
    parser = argparse.ArgumentParser(description="Dynamic LoRA adapter testing")
    parser.add_argument("--url", default=VLLM_URL, help="vLLM server URL")
    parser.add_argument("--swap-cycles", type=int, default=5,
                        help="Number of load/unload cycles for latency benchmark")
    parser.add_argument("--test", choices=["lifecycle", "swap", "all"], default="all",
                        help="Which test to run")
    args = parser.parse_args()

    global VLLM_URL
    VLLM_URL = args.url

    if args.test in ("lifecycle", "all"):
        test_dynamic_lifecycle()
    if args.test in ("swap", "all"):
        test_swap_latency(args.swap_cycles)


if __name__ == "__main__":
    main()
