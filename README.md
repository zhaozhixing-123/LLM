# nanochat D24 RunPod Serverless worker

This worker serves the nanochat SFT checkpoint trained at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

The model archive is not included in the image. Attach RunPod Network Volume
`99hnz378fr`; the worker reads:

`/runpod-volume/nanochat-d24-sft-inference.tar`

On the first startup it verifies SHA-256, extracts the archive to
`/runpod-volume/nanochat_artifacts`, and loads D24 step 466 on one CUDA GPU.
Later cold starts reuse the extracted files.

## Recommended endpoint settings

- GPU: 16 GB Ampere or newer for the first smoke test; use a 24 GB GPU if it OOMs
- Active workers: 0 minimum, 1 maximum
- Max concurrency: 1
- Idle timeout: 300 seconds for a classroom demonstration
- Execution timeout: 600 seconds
- Network Volume: `99hnz378fr`

## Request

```json
{
  "input": {
    "prompt": "What is the capital of France?",
    "max_new_tokens": 80,
    "temperature": 0.6,
    "top_k": 50,
    "seed": 42
  }
}
```

`messages` may be used instead of `prompt`. Messages must alternate between
`user` and `assistant`, start with `user`, and end with `user`.

## Security boundary

Do not expose a RunPod API key in browser JavaScript. The later Railway service
must keep the key in a server-side environment variable and proxy requests.
