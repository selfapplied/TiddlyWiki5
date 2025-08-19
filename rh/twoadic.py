#!/usr/bin/env python3
"""2-Adic Pyramid System - Multi-resolution culling for RH analysis"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import numpy as np


@dataclass
class TwoAdicPyramid:
    """2-adic pyramid for multi-resolution analysis"""
    base_length: int  # N = 2^m
    
    def __post_init__(self):
        if not (self.base_length & (self.base_length - 1) == 0):
            raise ValueError("base_length must be a power of 2")
        self.max_level = (self.base_length - 1).bit_length()
    
    def parent(self, i: int, level: int) -> int:
        """Parent at level ℓ: j = i >> ℓ"""
        return i >> level
    
    def child_range(self, j: int, level: int) -> Tuple[int, int]:
        """Children of j at level ℓ: [j2^ℓ, ..., j2^ℓ+2^ℓ-1]"""
        start = j << level
        end = start + (1 << level) - 1
        return start, end
    
    def wrap_index(self, i: int) -> int:
        """Wrap index mod N (circular)"""
        return i % self.base_length
    
    def build_pyramid_level(self, data: List[int], level: int, 
                           aggregator: Callable[[List[int]], int]) -> List[int]:
        """Build level ℓ+1 from level ℓ by pairing"""
        if level >= self.max_level:
            return data
        
        level_length = self.base_length >> level
        next_level_length = level_length >> 1
        
        result = []
        for j in range(next_level_length):
            left_idx = self.wrap_index(2 * j)
            right_idx = self.wrap_index(2 * j + 1)
            
            if left_idx < len(data) and right_idx < len(data):
                pair = [data[left_idx], data[right_idx]]
                result.append(aggregator(pair))
            elif left_idx < len(data):
                result.append(data[left_idx])
            else:
                result.append(0)
        
        return result
    
    def build_full_pyramid(self, data: List[int], 
                          aggregator: Callable[[List[int]], int]) -> List[List[int]]:
        """Build complete pyramid from base data"""
        pyramid = [data]
        
        for level in range(self.max_level):
            next_level = self.build_pyramid_level(pyramid[-1], level, aggregator)
            if len(next_level) > 0:
                pyramid.append(next_level)
        
        return pyramid


@dataclass
class TriMips:
    """Triangular multi-resolution pyramids for three unfoldings"""
    pyramid: TwoAdicPyramid
    
    def __init__(self, base_length: int):
        self.pyramid = TwoAdicPyramid(base_length)
    
    def build_tri_pyramids(self, mask: List[int], template: List[int]) -> Dict[str, List[List[int]]]:
        """Build three pyramids for rows, ↗, ↘ directions"""
        # Rows (horizontal)
        rows_pyramid = self.pyramid.build_full_pyramid(mask, sum)
        
        # ↗ diagonal (upper-right)
        diag_ur = self._extract_diagonal(mask, 1)
        diag_ur_pyramid = self.pyramid.build_full_pyramid(diag_ur, sum)
        
        # ↘ diagonal (lower-right)  
        diag_lr = self._extract_diagonal(mask, -1)
        diag_lr_pyramid = self.pyramid.build_full_pyramid(diag_lr, sum)
        
        return {
            "rows": rows_pyramid,
            "diag_ur": diag_ur_pyramid,
            "diag_lr": diag_lr_pyramid
        }
    
    def _extract_diagonal(self, data: List[int], direction: int) -> List[int]:
        """Extract diagonal with given direction"""
        N = len(data)
        diagonal = []
        
        for i in range(N):
            idx = (i * direction) % N
            diagonal.append(data[idx])
        
        return diagonal
    
    def compute_shift_bounds(self, mask_pyramids: Dict[str, List[List[int]]], 
                           template_pyramids: Dict[str, List[List[int]]], 
                           gap_threshold: int = 2) -> List[int]:
        """Compute shift bounds using Bloom-ish culling"""
        N = self.pyramid.base_length
        best_score = self._compute_best_score(mask_pyramids, template_pyramids)
        min_score = best_score - gap_threshold
        
        # Start with all shifts as candidates
        candidates = list(range(N))
        
        # Cull shifts from coarse to fine levels
        for level in range(len(mask_pyramids["rows"]) - 1, -1, -1):
            if len(candidates) <= 1:
                break
            
            survivors = []
            for shift in candidates:
                bound = self._compute_level_bound(mask_pyramids, template_pyramids, 
                                               level, shift)
                
                if bound >= min_score:
                    survivors.append(shift)
            
            candidates = survivors
        
        return candidates
    
    def _compute_best_score(self, mask_pyramids: Dict[str, List[List[int]]], 
                          template_pyramids: Dict[str, List[List[int]]]) -> int:
        """Compute best possible score for bound calculation"""
        mask_level = mask_pyramids["rows"][0]
        template_level = template_pyramids["rows"][0]
        
        max_mask = max(mask_level) if mask_level else 0
        max_template = max(template_level) if template_level else 0
        
        return max_mask * max_template
    
    def _compute_level_bound(self, mask_pyramids: Dict[str, List[List[int]]], 
                           template_pyramids: Dict[str, List[List[int]]], 
                           level: int, shift: int) -> int:
        """Compute U_ℓ(s) bound for given level and shift"""
        bound = 0
        
        # Sum bounds across all three directions
        for direction in ["rows", "diag_ur", "diag_lr"]:
            if level < len(mask_pyramids[direction]) and level < len(template_pyramids[direction]):
                mask_level = mask_pyramids[direction][level]
                template_level = template_pyramids[direction][level]
                
                level_bound = self._compute_direction_bound(mask_level, template_level, shift)
                bound += level_bound
        
        return bound
    
    def _compute_direction_bound(self, mask_level: List[int], template_level: List[int], 
                               shift: int) -> int:
        """Compute bound for one direction at given level"""
        bound = 0
        N = len(mask_level)
        
        for j in range(len(mask_level)):
            shifted_idx = (j + shift) % N
            if shifted_idx < len(template_level):
                bound += min(mask_level[j], template_level[shifted_idx])
        
        return bound


def demo_twoadic():
    """Quick demo of 2-adic pyramid system"""
    print("🔺 2-Adic Pyramid Demo")
    
    # Test with simple data
    N = 8
    mask = [1, 0, 1, 0, 1, 0, 1, 0]
    template = [1, 1, 0, 0, 1, 1, 0, 0]
    
    tri_mips = TriMips(N)
    mask_pyramids = tri_mips.build_tri_pyramids(mask, mask)
    template_pyramids = tri_mips.build_tri_pyramids(template, template)
    
    candidates = tri_mips.compute_shift_bounds(mask_pyramids, template_pyramids, gap_threshold=2)
    
    print(f"  • N = {N}, candidates after culling: {len(candidates)}")
    print(f"  • Pyramid levels: {len(mask_pyramids['rows'])}")
    print(f"  • Survivors: {candidates}")


if __name__ == "__main__":
    demo_twoadic()
