# Petals Inference Server

This directory provides a Docker configuration for running a Petals HTTP
inference server that connects to the decentralized Petals swarm (your
Raspberry Pi cluster).

## Configuration

Required environment variables:

- `HF_TOKEN` — HuggingFace token for gated models (e.g. Llama)
- `MODEL_NAME` — The model served by the swarm (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `INITIAL_PEERS` — Comma-separated list of initial peer multiaddrs to join the swarm

## Usage

On each Raspberry Pi, run a Petals server:

```bash
docker run -d \
  --name petals-pi-1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/bigscience-workshop/petals:main \
  python -m petals.cli.run_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num_blocks 8 \
    --initial_peers /ip4/<CENTRAL_IP>/tcp/31337/p2p/<PEER_ID>
```

The inference gateway runs in Docker Compose as part of the main stack
and connects to the swarm via the `INITIAL_PEERS` list.
