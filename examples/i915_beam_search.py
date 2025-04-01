#!/usr/bin/env python3
"""
Example demonstrating parallel beam search using the I915 backend.

This example shows how to use the I915 backend for parallel beam search in
a simple language model context. This showcases the benefit of having direct
control over the hardware through IOCTLs rather than using higher-level APIs.
"""

import argparse
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.device import Device
from tinygrad.helpers import getenv

def get_next_token_probs(input_ids, model):
    """Simplified token prediction. In reality, this would use a real LLM."""
    # Simulate getting next token probabilities from a model
    # This is a placeholder for actual model inference
    batch_size = input_ids.shape[0]
    vocab_size = 32000  # Common vocab size
    
    # Simulate some token prediction - in reality, this would use a real model
    probs = Tensor.randn(batch_size, vocab_size, device=input_ids.device)
    probs = probs.softmax(axis=1)  # Convert to probabilities
    
    return probs

def beam_search(model, input_ids, beam_width=4, max_length=20):
    """Perform beam search decoding using the I915 backend."""
    device = input_ids.device
    batch_size = input_ids.shape[0]
    vocab_size = 32000
    
    # Initial sequence is just the input_ids
    sequences = input_ids
    
    # Initial sequence scores (log probs)
    sequence_scores = Tensor.zeros(batch_size, 1, device=device)
    
    for _ in range(max_length):
        # Get next token probabilities
        next_token_probs = get_next_token_probs(sequences, model)
        next_token_logprobs = next_token_probs.log()
        
        # Calculate vocab scores for all beams
        vocab_scores = next_token_logprobs  # Shape: [batch_size, vocab_size]
        
        # Add previous scores to current vocabulary scores
        scores = sequence_scores + vocab_scores.reshape(batch_size, 1, vocab_size)
        scores = scores.reshape(batch_size, -1)  # Flatten to [batch_size, beam_width*vocab_size]
        
        # Get top-k scores and indices
        top_scores, top_indices = scores.topk(beam_width, axis=1)
        
        # Convert indices to vocab and beam indices
        beam_indices = top_indices // vocab_size
        vocab_indices = top_indices % vocab_size
        
        # Create new sequences
        new_sequences = []
        for b in range(batch_size):
            b_new_sequences = []
            for i in range(beam_width):
                beam_idx = beam_indices[b, i].numpy().item()
                vocab_idx = vocab_indices[b, i].numpy().item()
                
                b_new_sequences.append(
                    np.concatenate([sequences[b * beam_width + beam_idx].numpy(), [vocab_idx]])
                )
            new_sequences.append(np.stack(b_new_sequences))
        
        # Update sequences and scores
        sequences = Tensor(np.vstack(new_sequences), device=device)
        sequence_scores = top_scores
        
        # Check for completed sequences
        # In a real implementation, we'd check for EOS tokens and handle completed beams
    
    return sequences, sequence_scores

@TinyJit
def get_embedding(input_ids, embedding_matrix):
    """Get embeddings for input tokens."""
    # In a real implementation, this would be a lookup from an embedding matrix
    return embedding_matrix.take(input_ids)

def main():
    parser = argparse.ArgumentParser(description="I915 beam search example")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Check if I915 backend is enabled
    if not getenv("I915", 0):
        print("This example requires the I915 backend. Run with I915=1 python examples/i915_beam_search.py")
        return
    
    # Simplified model placeholder
    model = {"name": "beam_search_example"}
    
    # Create input tokens (batch_size=1 for simplicity)
    input_ids = Tensor([[101, 2054, 2003]], device="I915")  # [CLS] who is ...
    
    print(f"Running beam search with beam width {args.beam_width} and max length {args.max_length}...")
    
    # Run beam search
    sequences, scores = beam_search(model, input_ids, args.beam_width, args.max_length)
    
    # Print results
    print("\nTop completion sequences:")
    for i in range(args.beam_width):
        seq = sequences[i].numpy()
        score = scores[0, i].numpy().item()
        print(f"  Beam {i+1}: score={score:.4f}, sequence={seq}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()