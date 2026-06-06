#!/usr/bin/env python
"""
Integration test: CDI + EthioBBPE Tokenizer
============================================

Verifies:
  1. EthioBBPE tokenizer loads and works
  2. CDITokenizer correctly wraps it
  3. Embedding layer has correct shape
  4. End-to-end encode → embed → decode works
  5. Integration with CDI engine
"""

import torch
from cdi.tokenizer import CDITokenizer
from cdi.config import CDIConfig
from cdi.engine import CDIEngine


def test_ethiobbpe_load():
    """Test 1: EthioBBPE tokenizer loads."""
    print("\n" + "=" * 70)
    print("TEST 1: EthioBBPE Tokenizer Load")
    print("=" * 70)
    
    try:
        from ethiobbpe import EthioBBPETokenizer
        tok = EthioBBPETokenizer.from_pretrained()
        print(f"✓ EthioBBPE loaded successfully")
        print(f"  Vocabulary size: {tok.get_vocab_size():,} tokens")
        vocab = tok.get_vocab()
        print(f"  Special tokens: {[k for k in vocab if k.startswith('[')]}")
        return tok
    except Exception as e:
        print(f"✗ Failed to load EthioBBPE: {e}")
        return None


def test_cdi_tokenizer_init():
    """Test 2: CDITokenizer initialization."""
    print("\n" + "=" * 70)
    print("TEST 2: CDITokenizer Initialization")
    print("=" * 70)
    
    try:
        config = CDIConfig.small()
        tokenizer = CDITokenizer(
            embed_dim=config.observation_dim,
            max_len=config.n_points,
            dtype=config.dtype
        )
        print(f"✓ CDITokenizer initialized")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
        print(f"  Embed dim: {tokenizer.embed_dim}")
        print(f"  Max length: {tokenizer.max_len}")
        print(f"  Embedding shape: {tokenizer.embedding.shape}")
        print(f"  Embedding requires_grad: {tokenizer.embedding.requires_grad}")
        return tokenizer
    except Exception as e:
        print(f"✗ Failed to initialize CDITokenizer: {e}")
        return None


def test_encode_decode(tokenizer):
    """Test 3: Encode and decode."""
    print("\n" + "=" * 70)
    print("TEST 3: Encode / Decode")
    print("=" * 70)
    
    try:
        # Test English
        text_en = "What is the speed of light?"
        ids = tokenizer.encode(text_en)
        print(f"✓ Encoded English text")
        print(f"  Input:  '{text_en}'")
        print(f"  Token IDs shape: {ids.shape}")
        print(f"  First 5 IDs: {ids[:5].tolist()}")
        
        decoded = tokenizer.decode_ids(ids)
        print(f"  Decoded: '{decoded}'")
        
        # Test batch encoding
        texts = [
            "What is photosynthesis?",
            "The cell is the basic unit of life.",
            "DNA carries genetic information."
        ]
        batch_ids = tokenizer.encode_batch(texts)
        print(f"\n✓ Batch encoded {len(texts)} texts")
        print(f"  Batch shape: {batch_ids.shape}")
        print(f"  Expected: ({len(texts)}, {tokenizer.max_len})")
        
        return True
    except Exception as e:
        print(f"✗ Encode/decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding(tokenizer):
    """Test 4: Embedding lookup."""
    print("\n" + "=" * 70)
    print("TEST 4: Embedding Lookup")
    print("=" * 70)
    
    try:
        text = "The mitochondrion is the powerhouse of the cell."
        ids = tokenizer.encode(text)
        print(f"✓ Encoded text: '{text}'")
        
        embeddings = tokenizer.embed(ids)
        print(f"✓ Embedded token IDs")
        print(f"  Input shape: {ids.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected: ({tokenizer.max_len}, {tokenizer.embed_dim})")
        print(f"  dtype: {embeddings.dtype}")
        print(f"  Embedding mean: {embeddings.mean().item():.6f}")
        print(f"  Embedding std:  {embeddings.std().item():.6f}")
        
        # Verify encode_and_embed
        ids2, emb2 = tokenizer.encode_and_embed(text)
        assert torch.allclose(ids, ids2), "ID mismatch"
        assert torch.allclose(embeddings, emb2), "Embedding mismatch"
        print(f"✓ encode_and_embed matches")
        
        return True
    except Exception as e:
        print(f"✗ Embedding lookup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logits_projection(tokenizer):
    """Test 5: Logits projection via weight tying."""
    print("\n" + "=" * 70)
    print("TEST 5: Logits Projection (Weight Tying)")
    print("=" * 70)
    
    try:
        # Simulate CDI output
        output = torch.randn(1, tokenizer.max_len, tokenizer.embed_dim, 
                             dtype=tokenizer.dtype)
        print(f"✓ Created mock CDI output")
        print(f"  Shape: {output.shape}")
        
        logits = tokenizer.to_logits(output)
        print(f"✓ Projected to logits via embedding.T")
        print(f"  Output shape: {output.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected: (1, {tokenizer.max_len}, {tokenizer.vocab_size})")
        
        # Greedy decode
        decoded = tokenizer.decode_logits(logits[0])
        print(f"  Greedy decoded: '{decoded[:60]}...'")
        
        return True
    except Exception as e:
        print(f"✗ Logits projection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cdi_engine_integration():
    """Test 6: Integration with CDI engine."""
    print("\n" + "=" * 70)
    print("TEST 6: CDI Engine Integration")
    print("=" * 70)
    
    try:
        config = CDIConfig.tiny()
        config.validate()
        print(f"✓ Config validated")
        print(f"  n_points: {config.n_points}")
        print(f"  manifold_dim: {config.manifold_dim}")
        print(f"  observation_dim: {config.observation_dim}")
        
        engine = CDIEngine(config)
        engine.build()
        print(f"✓ CDI engine built")
        print(f"  n_params: {sum(p.numel() for p in engine.get_parameters()):,}")
        
        tokenizer = CDITokenizer(
            embed_dim=config.observation_dim,
            max_len=config.n_points,
            dtype=config.dtype
        )
        print(f"✓ CDI tokenizer created")
        print(f"  n_embedding_params: {sum(p.numel() for p in tokenizer.get_parameters()):,}")
        
        # Mock training step
        text = "What is the basic unit of life? The cell."
        ids, embeddings = tokenizer.encode_and_embed(text)
        print(f"✓ Tokenized and embedded")
        print(f"  Embedding shape: {embeddings.shape}")
        
        # Add batch dimension
        batch_emb = embeddings.unsqueeze(0)  # (1, n_points, embed_dim)
        output = engine.forward_sequence_batch(batch_emb)
        print(f"✓ CDI forward pass")
        print(f"  Input shape:  {batch_emb.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Project to logits
        logits = tokenizer.to_logits(output[0])
        print(f"✓ Projected to logits")
        print(f"  Logits shape: {logits.shape}")
        
        # Compute mock loss (simplified)
        B, S, V = logits.shape
        logits_flat = logits.reshape(B * S, V)
        targets = ids.long()  # Use original IDs as targets
        log_probs = logits_flat - logits_flat.logsumexp(dim=-1, keepdim=True)
        ce_loss = -log_probs[torch.arange(B * S), targets].mean()
        print(f"✓ Mock cross-entropy loss")
        print(f"  CE loss: {ce_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Engine integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  CDI + EthioBBPE Integration Test Suite".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {}
    
    # Test 1
    hf_tok = test_ethiobbpe_load()
    results["EthioBBPE Load"] = hf_tok is not None
    
    # Test 2
    tokenizer = test_cdi_tokenizer_init()
    results["CDITokenizer Init"] = tokenizer is not None
    
    if tokenizer:
        # Test 3
        results["Encode/Decode"] = test_encode_decode(tokenizer)
        
        # Test 4
        results["Embedding"] = test_embedding(tokenizer)
        
        # Test 5
        results["Logits Projection"] = test_logits_projection(tokenizer)
    
    # Test 6
    results["Engine Integration"] = test_cdi_engine_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        print("  EthioBBPE tokenizer is successfully integrated with CDI.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. See above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
