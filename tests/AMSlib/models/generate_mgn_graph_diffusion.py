#!/usr/bin/env python3
"""Train/export a tiny pure-Torch graph diffusion surrogate.

This file is meant to be readable as a small end-to-end example, even if you
have not worked with AI surrogate models before.

In this example, a "surrogate" is a learned replacement for a known
calculation. We first create synthetic graph data where the correct answer is
known exactly. Then we train a small neural network to imitate that answer.
Finally, we export the trained network and check that AMS/C++ gets the same
answer as Python/TorchScript for fixed test graphs.

The data is a graph:

* Nodes are sample points in a 2D square. Each node has features such as its
  x/y position, a scalar state value u, and a conductivity-like value kappa.
* Edges connect nearby nodes. Each edge has a source node and a destination
  node, so a message can travel "from source to destination".
* Global features describe the whole graph. Here the only global feature is
  dt, the time-step size.

The synthetic target is a diffusion update:

    delta_u_i = dt * sum_{src -> i} weight * (u_src - u_i)

Intuitively, each node is pulled toward its neighbors. If a neighboring source
node has a larger u value, it sends a positive contribution. If it has a
smaller u value, it sends a negative contribution. This is close in spirit to
heat diffusion, but cheap and fully deterministic.

The model is "MGN-like" (MeshGraphNet-like) because it repeats the same graph
communication pattern:

1. build/update an edge message from source node, destination node, edge data,
   and global data;
2. add incoming edge messages onto each destination node;
3. update each node from its old state and the aggregated incoming messages.

The workflow has three explicit modes:

* feasibility: prove that the model can run eagerly, be scripted, be reloaded,
  and still accept different graph sizes. This checks deployability before
  paying any training cost.
* train: train the tiny model on deterministic synthetic graphs, then save a
  checkpoint containing learned weights and metrics.
* fixtures: load the checkpoint, export the AMS-wrapped TorchScript model, and
  save fixed graph inputs plus Python TorchScript reference outputs.

The C++ parity test compares AMS output to Python TorchScript output, not to
the exact synthetic formula. Training already checks that the model learned the
synthetic target. The C++ test asks a different question: "when AMS deploys this
TorchScript graph model, does it reproduce Python inference?"
"""

import argparse
import copy
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for MGN graph diffusion generation. "
        "Install PyTorch or run this script on a Torch-enabled machine."
    ) from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from ams_model import create_ams_homogeneous_graph_model


NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 4
GLOBAL_FEATURE_DIM = 1
REFERENCE_OUTPUT_DIM = 1

FIXTURE_GRAPH_SIZES = (24, 73)
FEASIBILITY_SEEDS = (70024, 70073)
FIXTURE_SEEDS = (90024, 90073)

MODEL_SEED = 314159
TRAIN_DATA_SEED = 271828
VAL_DATA_SEED = 161803

LATENT_DIM = 64
NUM_PROCESSOR_BLOCKS = 2
K_NEIGHBORS = 6
TRAIN_STEPS = 3000
VAL_GRAPHS = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
PRIMARY_BASELINE_FACTOR = 0.1
SECONDARY_ABSOLUTE_MSE = 1.0e-4

CHECKPOINT_NAME = "mgn_graph_diffusion_checkpoint.pt"
MODEL_NAME = "mgn_graph_diffusion.pt"
FIXTURE_MANIFEST_NAME = "fixtures.json"
TRAINING_METRICS_NAME = "training_metrics.json"
FIXTURE_FORMAT_VERSION = 2
COMPARISON_RTOL = 2.0e-5
COMPARISON_ATOL = 2.0e-5

# Glossary for readers new to graph surrogates:
#
# N: number of nodes in one graph.
# E: number of directed edges in one graph.
# node_features: table with one row per node. Here [x, y, u, kappa].
# edge_index: integer connectivity table with shape [2, E]. Row 0 is source
#   node id, row 1 is destination node id.
# edge_features: table with one row per edge. Here [dx, dy, distance, message].
# global_features: graph-wide values. Here just [dt].
# delta_u: the node-wise output we want the model to predict.
# dt: time-step size in the synthetic diffusion update.
# kappa: conductivity-like node value used to define edge weights.
# checkpoint: saved training result containing model weights and metadata.
# fixture: fixed test input plus expected output used by C++ for repeatability.
# TorchScript: PyTorch's serialized model format that LibTorch/C++ can load.


class MLP(nn.Module):
    """A small trainable function used as the edge/node update rule.

    An MLP, or multilayer perceptron, is just a few linear layers with a
    nonlinearity. Here it learns simple formulas from examples instead of us
    hand-writing the formulas in C++.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ProcessorBlock(nn.Module):
    """One round of graph communication.

    A MeshGraphNet-style model alternates between updating edges and updating
    nodes. Think of this block as one "conversation round": edges look at their
    source and destination nodes, then nodes collect all incoming edge messages.
    """

    def __init__(self, latent_dim: int, global_dim: int):
        super().__init__()
        self.edge_mlp = MLP(latent_dim * 3 + global_dim, latent_dim, latent_dim)
        self.node_mlp = MLP(latent_dim * 2 + global_dim, latent_dim, latent_dim)

    def forward(
        self,
        node_latent: Tensor,
        edge_latent: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # AMS homogeneous graphs provide global_features as one graph-level
        # vector with shape [F_global]. For concatenation, this block needs one
        # copy per edge and one copy per node, so the broadcast happens here at
        # the point where graph-wide data is turned into per-entity inputs.
        global_row = global_features.unsqueeze(0)

        # The canonical AMS edge_index uses row 0 as source and row 1 as
        # destination. index_select gathers per-edge source/destination node
        # states without any graph-library dependency.
        src = edge_index[0]
        dst = edge_index[1]

        # Edge update: each edge sees its current latent state, source node,
        # destination node, and the graph-level dt feature. The residual update
        # keeps this tiny model easy to train.
        global_edges = global_row.expand(edge_latent.shape[0], global_row.shape[1])
        edge_input = torch.cat(
            (
                edge_latent,
                node_latent.index_select(0, src),
                node_latent.index_select(0, dst),
                global_edges,
            ),
            dim=1,
        )
        edge_latent = edge_latent + self.edge_mlp(edge_input)

        # Node aggregation: incoming edge states are summed onto destination
        # nodes with index_add, the same primitive TorchScript and LibTorch will
        # execute after export.
        aggregated = torch.zeros(
            (node_latent.shape[0], edge_latent.shape[1]),
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        aggregated = aggregated.index_add(0, dst, edge_latent)

        # Node update: each node sees its previous latent state, the aggregated
        # incoming message, and the global dt feature.
        global_nodes = global_row.expand(node_latent.shape[0], global_row.shape[1])
        node_input = torch.cat((node_latent, aggregated, global_nodes), dim=1)
        node_latent = node_latent + self.node_mlp(node_input)
        return node_latent, edge_latent


class TinyGraphDiffusionMGN(nn.Module):
    """Small MeshGraphNet-like model for the AMS graph contract.

    The structure is:

    * encoder: turn raw node/edge numbers into latent vectors the model can
      learn with;
    * processor blocks: let neighboring nodes exchange information through
      edges;
    * decoder: turn the final node latent vectors into node:delta_u.

    The model is intentionally tiny and fixed to two processor blocks. The goal
    is to validate the learned-model loop through AMS, not to build a large or
    configurable production architecture.
    """

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.node_encoder = MLP(node_dim, latent_dim, latent_dim)
        self.edge_encoder = MLP(edge_dim, latent_dim, latent_dim)
        self.processor1 = ProcessorBlock(latent_dim, global_dim)
        self.processor2 = ProcessorBlock(latent_dim, global_dim)
        self.node_decoder = MLP(latent_dim, latent_dim, REFERENCE_OUTPUT_DIM)

    def forward(self, graph: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Keep this signature and return type aligned with the AMS homogeneous
        # graph surrogate contract. The output key becomes outputs.node_fields
        # entry "delta_u" in C++.
        node_features = graph["node_features"]
        edge_index = graph["edge_index"].to(torch.int64)
        edge_features = graph["edge_features"]
        global_features = graph["global_features"]

        # Raw features such as x/y/u/kappa are low-dimensional physical-looking
        # values. Encoders map them into a latent space where the trainable
        # processor blocks can represent richer relationships.
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        # Two graph communication rounds are enough for this toy diffusion
        # problem: edges gather source/destination context, then destination
        # nodes receive aggregated incoming messages.
        node_latent, edge_latent = self.processor1(
            node_latent, edge_latent, edge_index, global_features
        )
        node_latent, edge_latent = self.processor2(
            node_latent, edge_latent, edge_index, global_features
        )

        # Decode one scalar per node. The "node:" prefix tells AMS this belongs
        # in outputs.node_fields rather than edge or global fields.
        delta_u = self.node_decoder(node_latent)
        return {"node:delta_u": delta_u}


def model_config() -> Dict[str, int]:
    return {
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "global_feature_dim": GLOBAL_FEATURE_DIM,
        "reference_output_dim": REFERENCE_OUTPUT_DIM,
        "latent_dim": LATENT_DIM,
        "num_processor_blocks": NUM_PROCESSOR_BLOCKS,
        "k_neighbors": K_NEIGHBORS,
    }


def make_model() -> TinyGraphDiffusionMGN:
    torch.manual_seed(MODEL_SEED)
    model = TinyGraphDiffusionMGN()
    return model.to(device=torch.device("cpu"), dtype=torch.float32)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _unique_directed_edges(knn: Tensor) -> Tensor:
    # kNN is computed from each node's perspective. Adding both directions makes
    # the fixture exercise directed source/destination semantics while still
    # representing an undirected diffusion neighborhood.
    pairs = set()
    num_nodes = int(knn.shape[0])
    for dst in range(num_nodes):
        for n in range(int(knn.shape[1])):
            src = int(knn[dst, n].item())
            if src == dst:
                continue
            pairs.add((src, dst))
            pairs.add((dst, src))
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.int64).t().contiguous()


def generate_graph(num_nodes: int, seed: int) -> Tuple[Dict[str, Tensor], Tensor]:
    """Create one deterministic synthetic graph and its exact diffusion target.

    This function is the "data generator" for the example. It replaces a real
    simulation code with a cheap formula that still has the graph ingredients we
    care about: node values, edge directions, edge weights, aggregation onto
    destination nodes, and a node-wise output field.
    """

    generator = make_generator(seed)

    # 1. Sample node locations in a unit square. In a mesh-based application
    # these would come from mesh vertices or degrees of freedom.
    positions = torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)

    # 2. Assign each node a smooth scalar state u. Smooth values are easier to
    # relate to heat diffusion than independent random noise: neighboring points
    # tend to have related values, but still differ enough to produce diffusion.
    phases = torch.rand((1, 4), generator=generator, dtype=torch.float32) * (2.0 * torch.pi)
    amps = torch.rand((1, 4), generator=generator, dtype=torch.float32) * 0.5 + 0.5
    x = positions[:, 0:1]
    y = positions[:, 1:2]
    u_raw = (
        amps[:, 0:1] * torch.sin(2.0 * torch.pi * x + phases[:, 0:1])
        + amps[:, 1:2] * torch.cos(2.0 * torch.pi * y + phases[:, 1:2])
        + amps[:, 2:3] * torch.sin(2.0 * torch.pi * (x + y) + phases[:, 2:3])
        + amps[:, 3:4] * torch.cos(2.0 * torch.pi * (x - y) + phases[:, 3:4])
    )
    u = torch.tanh(0.5 * u_raw)

    # 3. Add a conductivity-like scalar kappa and a graph-wide time step dt.
    # kappa changes how strongly values diffuse across edges; dt scales the
    # final update for every node in the graph.
    kappa = torch.rand((num_nodes, 1), generator=generator, dtype=torch.float32) + 0.5
    dt = torch.rand((1,), generator=generator, dtype=torch.float32) * 0.06 + 0.02

    # 4. Connect nearby nodes with a k-nearest-neighbor graph. The model sees
    # directed edges, so edge_index[0, e] is the source node and edge_index[1, e]
    # is the destination node for edge e. E varies with N because duplicate
    # edges are removed after adding the reverse direction; that helps test
    # dynamic node and edge counts in TorchScript and C++.
    distances = torch.cdist(positions, positions)
    masked = distances + torch.eye(num_nodes, dtype=torch.float32) * 1.0e6
    knn = torch.topk(masked, k=K_NEIGHBORS, largest=False, dim=1).indices
    edge_index = _unique_directed_edges(knn)
    src = edge_index[0]
    dst = edge_index[1]

    delta_pos = positions.index_select(0, src) - positions.index_select(0, dst)
    distance = delta_pos.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    conductivity = 0.5 * (kappa.index_select(0, src) + kappa.index_select(0, dst))
    raw_weight = conductivity / (distance + 0.05)

    # 5. Turn geometric distance and kappa into a diffusion weight. The incoming
    # weights for each destination node are normalized so every node receives a
    # bounded weighted average from its neighbors. This is only edge-weight
    # normalization in the synthetic formula, not model feature normalization.
    incoming_sum = torch.zeros((num_nodes, 1), dtype=torch.float32)
    incoming_sum = incoming_sum.index_add(0, dst, raw_weight)
    weight = raw_weight / incoming_sum.index_select(0, dst).clamp_min(1.0e-8)

    # 6. Build the tensors the model will receive. The fourth edge feature is
    # the weighted scalar message weight * (u_src - u_dst). Including it keeps
    # the learned problem small while still requiring source/destination indexing
    # and destination aggregation to recover node:delta_u.
    node_features = torch.cat((positions, u, kappa), dim=1).contiguous()
    messages = weight * (u.index_select(0, src) - u.index_select(0, dst))
    edge_features = torch.cat((delta_pos, distance, messages), dim=1).contiguous()

    # 7. Compute the exact supervised answer. For each edge, the source node
    # pushes the destination node toward u_src. index_add sums all incoming edge
    # messages for each destination node, then dt scales the update.
    target = torch.zeros((num_nodes, 1), dtype=torch.float32)
    target = target.index_add(0, dst, messages)
    target = dt * target

    graph = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "global_features": dt.contiguous(),
    }
    return graph, target.contiguous()


def graph_to_model_input(graph: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Canonicalize dtypes before eager, scripted, or reloaded model calls."""

    return {
        "node_features": graph["node_features"].to(dtype=torch.float32),
        "edge_index": graph["edge_index"].to(dtype=torch.int64),
        "edge_features": graph["edge_features"].to(dtype=torch.float32),
        "global_features": graph["global_features"].to(dtype=torch.float32),
    }


def assert_close(name: str, actual: Tensor, expected: Tensor, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        raise RuntimeError(f"{name} mismatch: max abs diff={max_diff}")


def validate_committed_fixture_manifest(fixture_dir: Path) -> None:
    """Ensure checked-in fixtures still describe this generator's contract."""

    manifest_path = fixture_dir / FIXTURE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing committed MGN graph diffusion manifest {manifest_path}. "
            "Regenerate and commit mgn_graph_diffusion.pt and fixtures.json."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def require_equal(name: str, actual: object, expected: object) -> None:
        if actual != expected:
            raise RuntimeError(
                f"Committed MGN fixture {name} is stale: "
                f"expected {expected!r}, found {actual!r}. Regenerate the fixtures."
            )

    model_record = manifest.get("model")
    if not isinstance(model_record, dict):
        raise RuntimeError("Committed MGN fixture model must be a JSON object. Regenerate the fixtures.")
    require_equal("format_version", manifest.get("format_version"), FIXTURE_FORMAT_VERSION)
    require_equal("model.path", model_record.get("path"), MODEL_NAME)

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("Committed MGN fixture metadata must be a JSON object. Regenerate the fixtures.")
    expected_metadata = {
        "model_config": model_config(),
        "model_seed": MODEL_SEED,
        "train_data_seed": TRAIN_DATA_SEED,
        "val_data_seed": VAL_DATA_SEED,
        "feasibility_seeds": list(FEASIBILITY_SEEDS),
        "fixture_graph_sizes": list(FIXTURE_GRAPH_SIZES),
        "fixture_seeds": list(FIXTURE_SEEDS),
    }
    for key, expected in expected_metadata.items():
        require_equal(f"metadata.{key}", metadata.get(key), expected)

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("Committed MGN fixture cases must be a JSON array. Regenerate the fixtures.")
    require_equal("case count", len(cases), len(FIXTURE_GRAPH_SIZES))
    for index, (case, num_nodes, seed) in enumerate(zip(cases, FIXTURE_GRAPH_SIZES, FIXTURE_SEEDS)):
        if not isinstance(case, dict):
            raise RuntimeError(f"Committed MGN fixture cases[{index}] must be a JSON object. Regenerate the fixtures.")
        expected_case = {
            "name": f"mgn_diffusion_n{num_nodes}",
            "seed": seed,
            "num_nodes": num_nodes,
            "node_feature_dim": NODE_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "reference_output_dim": REFERENCE_OUTPUT_DIM,
        }
        for key, expected in expected_case.items():
            require_equal(f"cases[{index}].{key}", case.get(key), expected)

    print(f"[info] Committed fixture metadata matches: {manifest_path}")


def run_feasibility(out_dir: Path, fixture_dir: Optional[Path] = None) -> None:
    # Mode 1: feasibility.
    #
    # Before training anything, prove that this model is deployable in the form
    # AMS needs. We run the same randomly initialized model four ways:
    #
    #   1. normal eager PyTorch;
    #   2. torch.jit.script output;
    #   3. the scripted model after saving and loading it;
    #   4. the AMS graph wrapper around the model.
    #
    # If these disagree, training would only hide a deployment problem. This
    # mode intentionally does not write fixtures or a trained checkpoint.
    out_dir.mkdir(parents=True, exist_ok=True)
    if fixture_dir is not None:
        validate_committed_fixture_manifest(fixture_dir)
    print("[info] MGN graph diffusion feasibility gate")
    print(f"[info] model_seed={MODEL_SEED}, fixture_sizes={FIXTURE_GRAPH_SIZES}, config={model_config()}")

    model = make_model().eval()
    graphs = [graph_to_model_input(generate_graph(n, seed)[0]) for n, seed in zip(FIXTURE_GRAPH_SIZES, FEASIBILITY_SEEDS)]

    with torch.no_grad():
        eager = [model(graph)["node:delta_u"].detach() for graph in graphs]

    scripted = torch.jit.script(model)
    with torch.no_grad():
        scripted_out = [scripted(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(scripted_out, eager)):
        assert_close(f"scripted graph {i}", actual, expected)

    script_path = out_dir / "mgn_graph_diffusion_feasibility.pt"
    scripted.save(str(script_path))
    reloaded = torch.jit.load(str(script_path)).eval()
    with torch.no_grad():
        reloaded_out = [reloaded(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(reloaded_out, eager)):
        assert_close(f"reloaded graph {i}", actual, expected)

    wrapped = create_ams_homogeneous_graph_model(model, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        wrapped_out = [wrapped(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(wrapped_out, eager)):
        assert_close(f"AMS-wrapped graph {i}", actual, expected)

    print(f"[info] Feasibility passed for dynamic graph sizes {FIXTURE_GRAPH_SIZES}")


def validation_graphs() -> List[Tuple[Dict[str, Tensor], Tensor]]:
    # Fixed validation seeds make the primary "10x better than zero baseline"
    # criterion reproducible across regeneration runs.
    graphs = []
    for i in range(VAL_GRAPHS):
        num_nodes = 16 + ((i * 37) % 113)
        graphs.append(generate_graph(num_nodes, VAL_DATA_SEED + i))
    return graphs


def evaluate(model: nn.Module, cases: Iterable[Tuple[Dict[str, Tensor], Tensor]]) -> Tuple[float, float, Tensor]:
    """Return model MSE, zero-prediction baseline MSE, and all targets."""

    total_loss = 0.0
    total_baseline = 0.0
    total_nodes = 0
    targets = []
    model.eval()
    with torch.no_grad():
        for graph, target in cases:
            pred = model(graph_to_model_input(graph))["node:delta_u"]
            loss_sum = (pred - target).pow(2).sum().item()
            baseline_sum = target.pow(2).sum().item()
            total_loss += loss_sum
            total_baseline += baseline_sum
            total_nodes += int(target.shape[0])
            targets.append(target)
    return total_loss / total_nodes, total_baseline / total_nodes, torch.cat(targets, dim=0)


def run_train(out_dir: Path) -> None:
    # Mode 2: train.
    #
    # The model sees many freshly generated synthetic graphs. For each graph we
    # know the exact target_delta_u, so training is ordinary supervised learning:
    # predict delta_u, measure mean-squared error, and update the network
    # weights. This mode writes learned weights and metrics only. It does not
    # export the AMS model or write the self-contained fixture manifest.
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[info] MGN graph diffusion training")
    print(f"[info] model_seed={MODEL_SEED}, train_data_seed={TRAIN_DATA_SEED}, val_data_seed={VAL_DATA_SEED}")
    print(f"[info] config={model_config()}")
    print(
        "[info] training "
        f"steps={TRAIN_STEPS}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"primary_pass=validation_mse <= {PRIMARY_BASELINE_FACTOR} * zero_baseline_mse"
    )

    model = make_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=(TRAIN_STEPS // 2, (TRAIN_STEPS * 5) // 6),
        gamma=0.3,
    )
    val_cases = validation_graphs()
    best_val_mse = float("inf")
    best_zero_baseline_mse = float("inf")
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    for step in range(1, TRAIN_STEPS + 1):
        # Generate one graph per optimization step. Avoiding batching keeps the
        # example small and makes dynamic graph sizes part of normal training.
        num_nodes = 16 + ((step * 53) % 113)
        graph, target = generate_graph(num_nodes, TRAIN_DATA_SEED + step)
        pred = model(graph_to_model_input(graph))["node:delta_u"]
        loss = torch.nn.functional.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 200 == 0 or step == TRAIN_STEPS:
            val_mse, zero_baseline_mse, _ = evaluate(model, val_cases)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_zero_baseline_mse = zero_baseline_mse
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())
            print(
                f"[info] step={step:04d} train_mse={loss.item():.8e} "
                f"val_mse={val_mse:.8e} zero_baseline_mse={zero_baseline_mse:.8e} "
                f"best_step={best_step:04d} best_val_mse={best_val_mse:.8e}"
            )

    model.load_state_dict(best_state_dict)
    val_mse, zero_baseline_mse, val_targets = evaluate(model, val_cases)
    target_mean = val_targets.mean().item()
    target_max_abs = val_targets.abs().max().item()
    target_std = val_targets.std(unbiased=False).item()
    primary_threshold = PRIMARY_BASELINE_FACTOR * zero_baseline_mse

    print(
        "[info] target_delta_u stats "
        f"mean={target_mean:.8e} max_abs={target_max_abs:.8e} std={target_std:.8e}"
    )
    print(
        f"[info] selected best_step={best_step}, validation_mse={val_mse:.8e}, "
        f"zero_baseline_mse={zero_baseline_mse:.8e}, primary_threshold={primary_threshold:.8e}"
    )
    # The primary success check is relative to a trivial model that always
    # predicts zero. That is easier to explain and more robust to target scale
    # changes than an absolute MSE alone.
    if val_mse > SECONDARY_ABSOLUTE_MSE:
        print(
            f"[warn] validation_mse={val_mse:.8e} is above secondary diagnostic "
            f"absolute MSE {SECONDARY_ABSOLUTE_MSE:.8e}"
        )
    if val_mse > primary_threshold:
        raise RuntimeError(
            "Training failed primary MGN graph diffusion criterion: "
            f"validation_mse={val_mse:.8e} > {primary_threshold:.8e}"
        )

    metadata = {
        "model_config": model_config(),
        "model_seed": MODEL_SEED,
        "train_data_seed": TRAIN_DATA_SEED,
        "val_data_seed": VAL_DATA_SEED,
        "feasibility_seeds": list(FEASIBILITY_SEEDS),
        "fixture_graph_sizes": list(FIXTURE_GRAPH_SIZES),
        "fixture_seeds": list(FIXTURE_SEEDS),
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "validation_mse": val_mse,
        "best_step": best_step,
        "best_validation_mse": best_val_mse,
        "best_zero_baseline_mse": best_zero_baseline_mse,
        "zero_baseline_mse": zero_baseline_mse,
        "primary_baseline_factor": PRIMARY_BASELINE_FACTOR,
        "secondary_absolute_mse": SECONDARY_ABSOLUTE_MSE,
        "target_mean": target_mean,
        "target_max_abs": target_max_abs,
        "target_std": target_std,
        "torch_version": str(torch.__version__),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    checkpoint_path = out_dir / CHECKPOINT_NAME
    torch.save(checkpoint, checkpoint_path)
    print(f"[info] Saved checkpoint: {checkpoint_path}")

    metrics_path = out_dir / TRAINING_METRICS_NAME
    metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[info] Saved training metrics: {metrics_path}")


def load_checkpoint(path: Path) -> Dict[str, object]:
    # PyTorch 2.6 changed torch.load's default weights_only behavior. The
    # checkpoint is generated by this local script and contains metadata, so
    # loading with weights_only=False is intentional here.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_manifest(tensor: Tensor, dtype: str) -> Dict[str, object]:
    """Return a self-contained row-major JSON record for one fixture tensor."""

    if dtype == "float32":
        packed_tensor = tensor.detach().cpu().to(dtype=torch.float32).contiguous()
        values = [float(x) for x in packed_tensor.view(-1).tolist()]
    elif dtype == "int64":
        packed_tensor = tensor.detach().cpu().to(dtype=torch.int64).contiguous()
        values = [int(x) for x in packed_tensor.view(-1).tolist()]
    else:
        raise ValueError(f"Unsupported fixture tensor dtype: {dtype}")

    return {
        "dtype": dtype,
        "shape": [int(dim) for dim in packed_tensor.shape],
        "values": values,
    }


def run_fixtures(out_dir: Path) -> None:
    # Mode 3: fixtures.
    #
    # A fixture is fixed test data: inputs plus the expected output. This mode
    # loads the trained checkpoint, exports the AMS-wrapped TorchScript model,
    # creates two fixed graph inputs, runs the exported model in Python, and
    # saves those model outputs as the C++ reference. Fixtures mode never
    # silently trains because parity tests should use an explicit checkpoint.
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Missing checkpoint {checkpoint_path}. Run --mode train, "
            "or `ctest -R MGN_DIFFUSION_TRAIN`, before --mode fixtures."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    print(f"[info] Loading checkpoint: {checkpoint_path}")
    print(f"[info] metadata={metadata}")

    model = make_model().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapped = create_ams_homogeneous_graph_model(model, torch.device("cpu"), torch.float32)

    model_path = out_dir / MODEL_NAME
    wrapped.save(str(model_path))
    print(f"[info] Saved AMS TorchScript model: {model_path}")

    # Reload the saved artifact before generating references. That catches
    # export-time issues and makes the Python reference match what C++ loads.
    reloaded = torch.jit.load(str(model_path)).eval()
    cases: List[Dict[str, object]] = []
    with torch.no_grad():
        for num_nodes, seed in zip(FIXTURE_GRAPH_SIZES, FIXTURE_SEEDS):
            graph, _target = generate_graph(num_nodes, seed)
            graph = graph_to_model_input(graph)
            output = reloaded(graph)["node:delta_u"].detach().cpu().contiguous()
            name = f"mgn_diffusion_n{num_nodes}"
            # The manifest embeds flattened row-major tensor values alongside
            # dtype and shape, keeping the checked-in fixture set to one JSON
            # file plus the TorchScript model.
            tensors = {
                "node_features": tensor_manifest(graph["node_features"], "float32"),
                "edge_index": tensor_manifest(graph["edge_index"], "int64"),
                "edge_features": tensor_manifest(graph["edge_features"], "float32"),
                "global_features": tensor_manifest(graph["global_features"], "float32"),
                "reference_delta_u": tensor_manifest(output, "float32"),
            }
            case = {
                "name": name,
                "seed": int(seed),
                "num_nodes": int(num_nodes),
                "num_edges": int(graph["edge_index"].shape[1]),
                "node_feature_dim": NODE_FEATURE_DIM,
                "edge_feature_dim": EDGE_FEATURE_DIM,
                "global_feature_dim": GLOBAL_FEATURE_DIM,
                "reference_output_dim": REFERENCE_OUTPUT_DIM,
                "tensors": tensors,
            }
            print(
                f"[info] fixture {case['name']} N={case['num_nodes']} "
                f"E={case['num_edges']} reference_shape={tuple(output.shape)}"
            )
            cases.append(case)

    # The manifest is self-contained except for the relative TorchScript path,
    # so model and JSON can be moved or committed as a two-file fixture set.
    manifest = {
        "format_version": FIXTURE_FORMAT_VERSION,
        "model": {
            "path": MODEL_NAME,
        },
        "metadata": metadata,
        "comparison": {
            "rtol": COMPARISON_RTOL,
            "atol": COMPARISON_ATOL,
        },
        "cases": cases,
    }
    manifest_path = out_dir / FIXTURE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[info] Saved fixture manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for MGN graph diffusion artifacts")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional committed fixture directory to validate during feasibility mode",
    )
    parser.add_argument(
        "--mode",
        choices=("feasibility", "train", "fixtures", "all"),
        required=True,
        help="Separated MGN graph diffusion generation mode",
    )
    args = parser.parse_args()

    if args.mode in ("feasibility", "all"):
        run_feasibility(args.out_dir, args.fixture_dir)
    if args.mode in ("train", "all"):
        run_train(args.out_dir)
    if args.mode in ("fixtures", "all"):
        run_fixtures(args.out_dir)


if __name__ == "__main__":
    main()
