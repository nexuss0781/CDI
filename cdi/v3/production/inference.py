"""Production inference and text generation engine for DCSS-CDI trained models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
import torch
import torch.nn.functional as F

from cdi.v3.tokenizer import CharacterTokenizer
from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import StageCConfig
from cdi.v3.production.checkpoints import load_verified
from cdi.v3.training import seed_everything


class DCSSInferenceEngine:
    """Inference wrapper for trained DCSS-CDI language models."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading inference engine from {checkpoint_path} on device {self.device}...")
        
        # Load verified atomic envelope checkpoint
        envelope = load_verified(checkpoint_path)
        payload = envelope["payload"]
        
        # Restore tokenizer
        tokenizer_artifact = payload["tokenizer_artifact"]
        self.tokenizer = CharacterTokenizer.from_artifact(tokenizer_artifact)
        
        # Restore model configuration and weights
        config_data = payload["config"]
        seed = config_data.get("seed", 42)
        seed_everything(seed)
        
        # Determine embedding dimension from model state dict
        model_state = payload["model_state"]
        embed_weight = model_state.get("token_embed.weight") or model_state.get("embedding.weight")
        if embed_weight is not None:
            embed_dim = embed_weight.shape[-1]
        else:
            embed_dim = 48  # Default spec alignment
            
        stage_c = StageCConfig.nano(seed=seed)
        stage_c = self._scale_stage_c(stage_c, embed_dim=embed_dim)
        
        self.model = DCSSLanguageModel(self.tokenizer, stage_c).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        print("Model loaded and ready for inference.")

    def _scale_stage_c(self, c_config: StageCConfig, embed_dim: int) -> StageCConfig:
        from dataclasses import replace
        return replace(
            c_config,
            input_width=embed_dim,
            output_width=embed_dim,
            n_vertices=4,
            band_width=4
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate text conditioned on a prompt using stateful selective recurrence."""
        self.model.eval()
        encoded = self.tokenizer.encode(prompt)
        input_ids = list(encoded.ids)
        
        if not input_ids:
            input_ids = [self.tokenizer.vocab.get("<bos>", 0)]

        curr_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            logits = self.model(curr_input)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < values[-1]] = float('-inf')
                
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            curr_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            
            if next_token == self.tokenizer.vocab.get("<eos>", -1):
                break
                
        from cdi.v3.tokenizer import EncodedText
        return self.tokenizer.decode(EncodedText(ids=tuple(generated), tokens=tuple(), text=""))


def interactive_chat(checkpoint_path: str | Path) -> None:
    """Run an interactive prompt generation loop in the terminal."""
    engine = DCSSInferenceEngine(checkpoint_path)
    print("\n" + "="*50)
    print("DCSS-CDI Interactive Inference CLI")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.strip().lower() in ("exit", "quit"):
                break
            if not prompt.strip():
                continue
                
            print("\nGenerating...")
            output = engine.generate(prompt, max_new_tokens=150, temperature=0.7, top_k=40)
            print(f"\nOutput:\n{output}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCSS-CDI Model Inference")
    parser.add_argument("--checkpoint", type=str, default="results/production/production_checkpoint.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for single generation")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()
    
    if args.prompt:
        engine = DCSSInferenceEngine(args.checkpoint)
        result = engine.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated:\n{result}")
    else:
        interactive_chat(args.checkpoint)
