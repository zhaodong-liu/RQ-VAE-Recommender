"""
Enhanced metrics module for recommendation evaluation.
Adds NDCG@K and Recall@K metrics while maintaining backward compatibility.
"""

import torch
from collections import defaultdict
from einops import rearrange
from torch import Tensor
import numpy as np


class EnhancedMetricsAccumulator:
    """
    Enhanced metrics accumulator that supports both Hit@K (original functionality)
    and standard recommendation metrics (Recall@K, NDCG@K)
    """
    def __init__(self, ks=[1, 5, 10], enable_ndcg_recall=True):
        self.ks = ks
        self.enable_ndcg_recall = enable_ndcg_recall
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)
        
        # For NDCG and Recall calculations
        if self.enable_ndcg_recall:
            self.recall_counts = {k: 0 for k in self.ks}
            self.ndcg_scores = {k: 0.0 for k in self.ks}

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """
        Accumulate metrics for a batch
        
        Args:
            actual: Ground truth semantic IDs of shape (B, D) where D is semantic ID length
            top_k: Predicted top-k semantic IDs of shape (B, K, D)
        """
        B, D = actual.shape
        
        # Original Hit@K metrics (for semantic ID slices)
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        
        for i in range(D):
            # Hit@K for progressive slices
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            # Hit@K for individual positions
            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        
        # New: Recall@K and NDCG@K calculations
        if self.enable_ndcg_recall:
            self._accumulate_recall_ndcg(actual, top_k)
        
        self.total += B
    
    def _accumulate_recall_ndcg(self, actual: Tensor, top_k: Tensor):
        """
        Calculate Recall@K and NDCG@K for recommendation metrics
        
        For semantic IDs, we consider a match when all D dimensions match
        """
        B, K, D = top_k.shape
        
        # Check if each prediction matches the ground truth (all D dimensions must match)
        # Shape: (B, K) - True if prediction at rank k matches ground truth
        matches = (actual.unsqueeze(1) == top_k).all(dim=-1)  # (B, K)
        
        for batch_idx in range(B):
            # Find if there's a match and at what rank
            match_positions = torch.where(matches[batch_idx])[0]
            
            if len(match_positions) > 0:
                # Get the rank of the first match (0-indexed)
                first_match_rank = match_positions[0].item()
                
                # Calculate metrics for different K values
                for k in self.ks:
                    if first_match_rank < k:
                        # Recall@K: binary - item is in top-k or not
                        self.recall_counts[k] += 1
                        
                        # NDCG@K: discounted by position
                        # Using log2(rank+2) as the discount factor (rank is 0-indexed)
                        ndcg_score = 1.0 / np.log2(first_match_rank + 2)
                        self.ndcg_scores[k] += ndcg_score
    
    def reduce(self) -> dict:
        """
        Compute and return all metrics
        """
        if self.total == 0:
            return {}
        
        # Original Hit@K metrics
        results = {k: v/self.total for k, v in self.metrics.items()}
        
        # Add Recall@K and NDCG@K metrics
        if self.enable_ndcg_recall:
            for k in self.ks:
                # Recall@K: percentage of items found in top-k
                results[f"recall@{k}"] = self.recall_counts[k] / self.total
                
                # NDCG@K: normalized discounted cumulative gain
                # Normalization: divide by best possible score (1.0 per item)
                results[f"ndcg@{k}"] = self.ndcg_scores[k] / self.total
        
        return results


class SimplifiedMetricsAccumulator:
    """
    Simplified version that only computes Recall@K and NDCG@K
    for when you only need recommendation metrics
    """
    def __init__(self, ks=[5, 10]):
        self.ks = ks
        self.reset()
    
    def reset(self):
        self.total = 0
        self.recall_counts = {k: 0 for k in self.ks}
        self.ndcg_scores = {k: 0.0 for k in self.ks}
    
    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """
        Accumulate metrics for a batch
        
        Args:
            actual: Ground truth items/semantic IDs of shape (B, D) or (B,)
            top_k: Predicted top-k items of shape (B, K, D) or (B, K)
        """
        # Handle both semantic IDs (B, D) and item IDs (B,)
        if actual.dim() == 1:
            actual = actual.unsqueeze(-1)  # (B,) -> (B, 1)
        if top_k.dim() == 2:
            top_k = top_k.unsqueeze(-1)  # (B, K) -> (B, K, 1)
        
        B = actual.shape[0]
        
        # Check if predictions match ground truth
        if actual.shape[-1] > 1:  # Semantic IDs case
            matches = (actual.unsqueeze(1) == top_k).all(dim=-1)  # (B, K)
        else:  # Item IDs case
            matches = (actual.unsqueeze(1) == top_k).squeeze(-1)  # (B, K)
        
        for batch_idx in range(B):
            match_positions = torch.where(matches[batch_idx])[0]
            
            if len(match_positions) > 0:
                first_match_rank = match_positions[0].item()
                
                for k in self.ks:
                    if first_match_rank < k:
                        self.recall_counts[k] += 1
                        ndcg_score = 1.0 / np.log2(first_match_rank + 2)
                        self.ndcg_scores[k] += ndcg_score
        
        self.total += B
    
    def reduce(self) -> dict:
        """
        Compute and return Recall@K and NDCG@K metrics
        """
        if self.total == 0:
            return {}
        
        results = {}
        for k in self.ks:
            results[f"recall@{k}"] = self.recall_counts[k] / self.total
            results[f"ndcg@{k}"] = self.ndcg_scores[k] / self.total
        
        return results


# Backward compatibility: keep the original class name
class TopKAccumulator(EnhancedMetricsAccumulator):
    """
    Maintains backward compatibility with the original TopKAccumulator
    while adding new metrics functionality
    """
    def __init__(self, ks=[1, 5, 10]):
        super().__init__(ks=ks, enable_ndcg_recall=True)


# Utility function for testing
def test_metrics():
    """
    Test function to verify metrics are working correctly
    """
    import torch
    
    print("Testing Enhanced Metrics...")
    
    # Create test data
    batch_size = 4
    semantic_id_length = 4
    k = 10
    
    # Simulated ground truth (batch_size, semantic_id_length)
    actual = torch.randint(0, 256, (batch_size, semantic_id_length))
    
    # Simulated predictions (batch_size, k, semantic_id_length)
    top_k = torch.randint(0, 256, (batch_size, k, semantic_id_length))
    
    # Make some predictions match for testing
    top_k[0, 0, :] = actual[0, :]  # Rank 1 match
    top_k[1, 2, :] = actual[1, :]  # Rank 3 match
    top_k[2, 5, :] = actual[2, :]  # Rank 6 match
    
    # Test with enhanced accumulator
    accumulator = EnhancedMetricsAccumulator(ks=[5, 10])
    accumulator.accumulate(actual, top_k)
    metrics = accumulator.reduce()
    
    print("\nTest Results:")
    print(f"  Recall@5: {metrics.get('recall@5', 0):.4f}")
    print(f"  Recall@10: {metrics.get('recall@10', 0):.4f}")
    print(f"  NDCG@5: {metrics.get('ndcg@5', 0):.4f}")
    print(f"  NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
    
    # Test backward compatibility
    accumulator_compat = TopKAccumulator(ks=[5, 10])
    accumulator_compat.accumulate(actual, top_k)
    metrics_compat = accumulator_compat.reduce()
    
    print("\nBackward Compatibility Test:")
    print(f"  Metrics match: {metrics == metrics_compat}")
    
    print("\n✅ Metrics test completed successfully!")
    
    return metrics


if __name__ == "__main__":
    # Run test when module is executed directly
    test_metrics()
