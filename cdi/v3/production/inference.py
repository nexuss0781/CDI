"""Fail-closed, stateful inference for verified DCSS-CDI production checkpoints.

This module never mutates a checkpoint.  It reconstructs the tokenizer and model
from the verified production envelope, checks all binding fingerprints, and
performs causal generation through ``DCSSLanguageModel.forward_chunk`` with a
carried cohomodynamic state.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.production.checkpoints import load_verified
from cdi.v3.ssm import CohomodynamicState, StageCConfig
from cdi.v3.tokenizer import CharacterTokenizer, TokenizerConfig
from cdi.v3.training import parameter_fingerprint


SamplingMode = Literal["greedy", "sample"]


@dataclass(frozen=True)
class GenerationConfig:
    """Validated decoding controls for a single causal generation request."""

    max_new_tokens: int = 128
    mode: SamplingMode = "sample"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 42
    stop_at_eos: bool = True
    max_prompt_tokens: int = 2048

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if self.mode not in ("greedy", "sample"):
            raise ValueError("mode must be 'greedy' or 'sample'.")
        if not 0.0 < self.temperature <= 5.0:
            raise ValueError("temperature must be in (0, 5].")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive.")


@dataclass(frozen=True)
class InferenceMetadata:
    """Immutable checkpoint identity exposed after successful restoration."""

    checkpoint_path: str
    checkpoint_sha256: str
    lineage_fingerprint: str
    tokenizer_fingerprint: str
    model_fingerprint: str
    topology_fingerprint: str | None
    global_step: int
    device: str
    dtype: str


def _tokenizer_from_artifact(artifact: Mapping[str, Any]) -> CharacterTokenizer:
    """Restore and independently validate the tokenizer artifact stored in a checkpoint."""
    if artifact.get("format") != TokenizerConfig().format:
        raise ValueError("Checkpoint contains an unsupported tokenizer artifact format.")
    if "config" not in artifact or "vocabulary" not in artifact or "fingerprint" not in artifact:
        raise ValueError("Checkpoint tokenizer artifact is incomplete.")
    if artifact.get("fingerprint") != CharacterTokenizer._fingerprint_payload(artifact):
        raise ValueError("Checkpoint tokenizer artifact fingerprint does not match its contents.")
    config_data = dict(artifact["config"])
    config_data["special_tokens"] = tuple(config_data["special_tokens"])
    tokenizer = CharacterTokenizer(list(artifact["vocabulary"]), TokenizerConfig(**config_data))
    tokenizer.assert_fingerprint(str(artifact["fingerprint"]))
    return tokenizer


def _state_tensor(state: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        value = state[name]
    except KeyError as exc:
        raise ValueError(f"Checkpoint model_state lacks required tensor {name!r}.") from exc
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint model_state entry {name!r} is not a tensor.")
    return value


def _reconstruct_stage_c_config(model_state: Mapping[str, torch.Tensor], seed: int, device: str) -> StageCConfig:
    """Infer the supported structured architecture from checkpoint tensor shapes.

    The model embeds the input width in ``embedding.weight`` and stores one
    ``input_injection`` tensor per memory band with shape
    ``(n_vertices, band_width)``.  This permits strict restoration of both the
    current 4-by-4 production configuration and valid earlier nano checkpoints
    without guessing their state dimensions.
    """
    embedding = _state_tensor(model_state, "embedding.weight")
    if embedding.ndim != 2 or embedding.shape[0] <= 0 or embedding.shape[1] <= 0:
        raise ValueError("Checkpoint embedding.weight must have positive rank-2 shape.")
    if embedding.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Checkpoint embedding dtype {embedding.dtype} is unsupported.")
    shape_reference = _state_tensor(model_state, "ssm.cell.bands.fast.generator.input_injection")
    if shape_reference.ndim != 2 or shape_reference.shape[0] < 3 or shape_reference.shape[1] <= 0:
        raise ValueError("Checkpoint fast-band input_injection must have shape (n_vertices >= 3, band_width > 0).")
    n_vertices, band_width = (int(shape_reference.shape[0]), int(shape_reference.shape[1]))
    if band_width % 2:
        raise ValueError("Checkpoint band_width must be even for the stable pairwise-skew integrator.")
    for band in ("fast", "middle", "harmonic"):
        injection = _state_tensor(model_state, f"ssm.cell.bands.{band}.generator.input_injection")
        timescale = _state_tensor(model_state, f"ssm.cell.bands.{band}.generator.log_tau_base")
        if tuple(injection.shape) != (n_vertices, band_width) or tuple(timescale.shape) != (band_width,):
            raise ValueError(f"Checkpoint {band} memory-band shapes are inconsistent with the structured state.")
    readout = _state_tensor(model_state, "ssm.cell.readout.weight")
    expected_readout_shape = (int(embedding.shape[1]), 3 * band_width)
    if tuple(readout.shape) != expected_readout_shape:
        raise ValueError(
            "Checkpoint readout shape is inconsistent with its embedding width and structured state: "
            f"expected {expected_readout_shape}, found {tuple(readout.shape)}."
        )
    dtype_str = "float32" if embedding.dtype == torch.float32 else "float64"
    base = StageCConfig.nano(seed=seed)
    config = replace(
        base,
        input_width=int(embedding.shape[1]),
        output_width=int(embedding.shape[1]),
        n_vertices=n_vertices,
        band_width=band_width,
        dtype_str=dtype_str,
        device=device,
    )
    config.validate()
    return config


class DCSSInferenceEngine:
    """Stateful DCSS-CDI generation from a verified production checkpoint.

    Construction validates the atomic checkpoint sidecar, envelope, tokenizer,
    strict model state dictionary, topology fingerprint, and trained model
    fingerprint.  Any mismatch rejects loading rather than silently guessing.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable.")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device string.")
        self.device = requested
        self.checkpoint_path = Path(checkpoint_path)

        envelope = load_verified(self.checkpoint_path)
        payload = envelope.get("stage_d_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Verified production envelope lacks a Stage D payload.")
        if payload.get("format") != "dcss-cdi-stage-d-checkpoint-v1":
            raise ValueError("Unsupported Stage D payload format for inference.")

        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("Checkpoint Stage D payload lacks model_state.")
        tokenizer_artifact = payload.get("tokenizer_artifact")
        if not isinstance(tokenizer_artifact, Mapping):
            raise ValueError("Checkpoint Stage D payload lacks tokenizer_artifact.")

        self.tokenizer = _tokenizer_from_artifact(tokenizer_artifact)
        if payload.get("tokenizer_fingerprint") != self.tokenizer.fingerprint:
            raise ValueError("Checkpoint tokenizer fingerprint does not match restored tokenizer.")
        lineage = envelope.get("lineage", {})
        if not isinstance(lineage, Mapping) or lineage.get("tokenizer_fingerprint") != self.tokenizer.fingerprint:
            raise ValueError("Checkpoint lineage tokenizer binding is invalid.")

        embedding = _state_tensor(model_state, "embedding.weight")
        if int(embedding.shape[0]) != self.tokenizer.vocab_size:
            raise ValueError("Checkpoint embedding vocabulary size does not match tokenizer vocabulary.")
        if int(self.tokenizer.config.embedding_dim) != int(embedding.shape[1]):
            raise ValueError("Checkpoint tokenizer embedding width does not match model embedding width.")
        output_bias = _state_tensor(model_state, "output_bias")
        if tuple(output_bias.shape) != (self.tokenizer.vocab_size,):
            raise ValueError("Checkpoint output bias shape does not match tokenizer vocabulary.")

        config_data = payload.get("config", {})
        if not isinstance(config_data, Mapping):
            raise ValueError("Checkpoint training configuration is invalid.")
        seed = int(config_data.get("seed", 42))
        self.stage_c_config = _reconstruct_stage_c_config(model_state, seed=seed, device=self.device)
        self.model = DCSSLanguageModel(self.tokenizer, self.stage_c_config).to(self.device)
        try:
            self.model.load_state_dict(dict(model_state), strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint is incompatible with the supported DCSS-CDI architecture: {exc}") from exc
        self.model.eval()

        expected_topology = payload.get("topology_fingerprint")
        actual_topology = self.model.ssm.cell.topology.fingerprint()
        if expected_topology is not None and expected_topology != actual_topology:
            raise ValueError("Checkpoint topology fingerprint does not match reconstructed topology.")
        actual_model_fingerprint = parameter_fingerprint(self.model)
        if lineage.get("model_fingerprint") != actual_model_fingerprint:
            raise ValueError("Checkpoint lineage model fingerprint does not match restored model state.")

        sidecar = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".sha256")
        sidecar_fields = dict(
            line.split("=", 1)
            for line in sidecar.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        self.metadata = InferenceMetadata(
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=str(sidecar_fields["sha256"]),
            lineage_fingerprint=str(envelope["lineage_fingerprint"]),
            tokenizer_fingerprint=self.tokenizer.fingerprint,
            model_fingerprint=actual_model_fingerprint,
            topology_fingerprint=expected_topology,
            global_step=int(payload.get("global_step", 0)),
            device=self.device,
            dtype=self.stage_c_config.dtype_str,
        )

    def _prepare_prompt(self, prompt: str, max_prompt_tokens: int) -> torch.Tensor:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [self.tokenizer.bos_id, *encoded.ids]
        if len(tokens) > max_prompt_tokens:
            raise ValueError(
                f"Prompt encodes to {len(tokens)} tokens, exceeding max_prompt_tokens={max_prompt_tokens}. "
                "Truncate the prompt explicitly before calling generate()."
            )
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def _filtered_logits(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        candidate = logits.detach().clone().to(dtype=torch.float32)
        forbidden = (self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.doc_id)
        candidate[list(forbidden)] = float("-inf")
        if not torch.isfinite(candidate).any():
            raise RuntimeError("No valid token remains after special-token suppression.")
        if config.mode == "greedy":
            return candidate
        candidate = candidate / config.temperature
        if config.top_k:
            keep = min(config.top_k, candidate.numel())
            threshold = torch.topk(candidate, keep).values[-1]
            candidate[candidate < threshold] = float("-inf")
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(candidate, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) > config.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            candidate[sorted_indices[remove]] = float("-inf")
        if not torch.isfinite(candidate).any():
            raise RuntimeError("No valid token remains after sampling filters.")
        return candidate

    @torch.inference_mode()
    def generate_ids(self, prompt: str, config: GenerationConfig | None = None) -> torch.Tensor:
        """Return prompt and newly generated token IDs using a carried recurrent state."""
        generation = config or GenerationConfig()
        generation.validate()
        prefix = self._prepare_prompt(prompt, generation.max_prompt_tokens)
        logits, state = self.model.forward_chunk(prefix, return_state=True)
        generated = prefix[0].tolist()
        next_logits = logits[0, -1]
        rng = torch.Generator(device=self.device).manual_seed(generation.seed)

        for _ in range(generation.max_new_tokens):
            candidate = self._filtered_logits(next_logits, generation)
            if generation.mode == "greedy":
                next_token = int(torch.argmax(candidate).item())
            else:
                probabilities = torch.softmax(candidate, dim=-1)
                next_token = int(torch.multinomial(probabilities, num_samples=1, generator=rng).item())
            generated.append(next_token)
            if generation.stop_at_eos and next_token == self.tokenizer.eos_id:
                break
            step_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, state = self.model.forward_chunk(step_ids, state=state, return_state=True)
            next_logits = logits[0, -1]
        return torch.tensor(generated, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate decoded continuation text, including the supplied prompt."""
        return self.tokenizer.decode(self.generate_ids(prompt, config))

    @torch.inference_mode()
    def complete(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate only the continuation, excluding the normalized prompt prefix."""
        output = self.generate(prompt, config)
        normalized = self.tokenizer.normalize(prompt)
        return output[len(normalized):] if output.startswith(normalized) else output


def interactive_chat(checkpoint_path: str | Path, device: str | None = None) -> None:
    """Run a minimal interactive local inference loop; no tools or side effects occur."""
    engine = DCSSInferenceEngine(checkpoint_path, device=device)
    print("DCSS-CDI interactive inference; type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("Prompt > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip().lower() in {"exit", "quit"}:
            return
        if not prompt.strip():
            continue
        try:
            print(engine.complete(prompt, GenerationConfig(max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.95)))
        except Exception as exc:
            print(f"Generation rejected: {exc}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verified DCSS-CDI checkpoint inference")
    parser.add_argument("--checkpoint", default="results/production/production_checkpoint.pt")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    engine = DCSSInferenceEngine(args.checkpoint, device=args.device)
    if args.prompt is None:
        interactive_chat(args.checkpoint, device=args.device)
        return
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        mode="greedy" if args.greedy else "sample",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(engine.generate(args.prompt, config))


if __name__ == "__main__":
    _main()
