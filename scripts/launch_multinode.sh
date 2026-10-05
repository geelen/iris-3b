#!/usr/bin/env bash
# Multi-node training: ONE torchrun rendezvous spanning every node.
#
# Run this on every node with the same NNODES / RDZV endpoint and a distinct
# NODE_RANK. Do NOT fall back to `torchrun --nproc_per_node=8` per node: that
# builds one independent world per node, each with its own rank 0, so both
# halves walk the same data range and both write checkpoints, with no error.
# `--nnodes=$NNODES` is a fixed count, so the c10d rendezvous blocks until every
# node has joined and a half-formed job cannot start at all.
#
# Required:  NNODES, NODE_RANK, and MASTER_ADDR (or RDZV_ENDPOINT=host:port)
# Optional:  MASTER_PORT (29500), NPROC (8), IRIS_ROOT (the checkout holding
#            this script), IRIS_LOG (output/multinode.node$NODE_RANK.log under
#            IRIS_ROOT; "-" keeps stdout),
#            RDZV_ID (defaults from the config name; two jobs running at once
#            need distinct ids or they join each other's rendezvous),
#            NODE_LIST, NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_DEBUG_INFO, PRESTAGE
#
#   NNODES=8 NODE_RANK=0 MASTER_ADDR=node0 NCCL_SOCKET_IFNAME=eth0 \
#     scripts/launch_multinode.sh configs/iris3b/stage1_256.yaml data.data_dirs=[/path/to/wids/dataset]
set -euo pipefail

ROOT=${IRIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

CONFIG=${1:?usage: NNODES=n NODE_RANK=i MASTER_ADDR=host scripts/launch_multinode.sh CONFIG [overrides...]}
shift
NNODES=${NNODES:?NNODES is required; every node must pass the same value}
NODE_RANK=${NODE_RANK:?NODE_RANK is required and must be distinct per node (0..NNODES-1)}
NPROC=${NPROC:-8}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-${MASTER_ADDR:?MASTER_ADDR or RDZV_ENDPOINT is required}:$MASTER_PORT}
RDZV_HOST=${RDZV_ENDPOINT%%:*}
RDZV_PORT=${RDZV_ENDPOINT##*:}
case "$RDZV_PORT" in '' | *[!0-9]*) RDZV_PORT=$MASTER_PORT ;; esac
RDZV_ID=${RDZV_ID:-iris-$(basename "$CONFIG" .yaml)}

cd "$ROOT"
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi

# `--node-rank` is honoured only by the static rendezvous backend; with c10d the
# rendezvous assigns GROUP_RANK by sorted participant FQDN. That matters because
# the pre-staged shape-plan caches are keyed by global rank. Give NODE_LIST (the
# job's FQDNs, comma separated) to check the operator's NODE_RANK against the
# order c10d will actually pick, instead of assuming they agree.
if [ -n "${NODE_LIST:-}" ]; then
  probe=$(NODE_LIST="$NODE_LIST" python - <<'PY'
import os
import socket

nodes = sorted(n.strip() for n in os.environ["NODE_LIST"].split(",") if n.strip())
me = socket.getfqdn()
print(nodes.index(me) if me in nodes else -1, me)
PY
)
  derived=${probe%% *}
  if [ "$derived" != "$NODE_RANK" ]; then
    echo "ABORT: NODE_RANK=$NODE_RANK but this host sorts at index $derived in NODE_LIST" >&2
    echo "       (index -1 means socket.getfqdn() '${probe#* }' is not in NODE_LIST)" >&2
    exit 1
  fi
fi

# Fabric selection is the only NCCL tuning that is safe to pin: it picks which
# interface to use, not how a collective is computed.
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then export NCCL_SOCKET_IFNAME; fi
# The leading '=' is mandatory. Without it the value is a PREFIX match, so
# "mlx5_0" also selects mlx5_0x and every other device sharing that prefix.
if [ -n "${NCCL_IB_HCA:-}" ]; then export NCCL_IB_HCA="=${NCCL_IB_HCA#=}"; fi
if [ -n "${NCCL_DEBUG_INFO:-}" ]; then export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV; fi
# Never set NCCL_ALGO, NCCL_PROTO or NCCL_BUFFSIZE. Forcing an algorithm or
# protocol, or resizing the transport buffers, bypasses NCCL's own capability
# checks and can silently corrupt a reduction. Unset here so an inherited
# export cannot leak into the job.
unset NCCL_ALGO NCCL_PROTO NCCL_BUFFSIZE

# Read by scripts/train.py as the declared node count even if --expected-nodes
# is dropped from the command line.
export NNODES

LOG=${IRIS_LOG:-output/multinode.node$NODE_RANK.log}
echo "node $NODE_RANK/$NNODES x $NPROC gpu -> rdzv $RDZV_HOST:$RDZV_PORT (id $RDZV_ID), log $LOG"
if [ "$LOG" != "-" ]; then
  mkdir -p "$(dirname "$LOG")"
  exec >>"$LOG" 2>&1
fi
echo "=== node $NODE_RANK launch $(date -u), root $ROOT"

# One process per node fills the caches that are populated lazily on first use
# and are node-local: the torch.hub REPA teacher (concurrent loads race on
# extraction) and this node's shape-plan key caches. Ranks on node 1+ otherwise
# all start cold at once. Needs the same env the workers will see, so the
# topology is checked here too, before any GPU is claimed.
if [ "${PRESTAGE:-0}" = 1 ]; then
  RANK=$((NODE_RANK * NPROC)) LOCAL_RANK=0 WORLD_SIZE=$((NNODES * NPROC)) \
    LOCAL_WORLD_SIZE="$NPROC" GROUP_RANK="$NODE_RANK" GROUP_WORLD_SIZE="$NNODES" \
    MASTER_ADDR="$RDZV_HOST" MASTER_PORT="$RDZV_PORT" \
    python scripts/train.py --prestage --config "$CONFIG" --expected-nodes "$NNODES" "$@"
fi

exec torchrun \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --nproc-per-node="$NPROC" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
  --rdzv-id="$RDZV_ID" \
  scripts/train.py --config "$CONFIG" --expected-nodes "$NNODES" "$@"
