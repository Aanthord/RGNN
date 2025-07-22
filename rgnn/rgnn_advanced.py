"""
RGNN Advanced Implementation
===========================

Recursive Graded Neural Networks based on Recursive Shape-Structured Notation (RSSN)
and fractal density functions. This module implements the theoretical framework from
the RGNN paper in practical PyTorch neural networks.

Author: Michael A. Doran Jr.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import warnings


class RecursiveDensityFunction(nn.Module):
    """
    Learnable fractal density function D_k(n) that captures recursive structure
    at different scales. This is the core of the RGNN framework.
    
    The density function approximates:
    D_k(n) = lim_{i→∞} F_i(n) / G_i
    
    Where F_i counts recursive substructures and G_i is the configuration space.
    """
    
    def __init__(self, input_dim: int, max_depth: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        
        # Multi-layer network to approximate the complex density function
        self.density_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for recursion level
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Density ∈ [0,1]
        )
        
        # Theoretical mode: use RSSN formulas directly
        self.theoretical_mode = False
        
    def enable_theoretical_mode(self):
        """Enable theoretical RSSN density calculations"""
        self.theoretical_mode = True
        
    def disable_theoretical_mode(self):
        """Use learned density approximation"""
        self.theoretical_mode = False
        
    def compute_theoretical_density(self, x: torch.Tensor, recursion_level: int) -> torch.Tensor:
        """
        Compute density using theoretical RSSN formulas
        For basic operators: D_k(n) ≈ 1/n
        """
        # Compute characteristic value from input
        n_values = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        n_values = torch.clamp(n_values, min=1.0)  # Avoid division by zero
        
        # Basic RSSN density formula with level adjustment
        base_density = 1.0 / n_values
        level_adjustment = 1.0 / (recursion_level + 1)
        
        density = base_density * level_adjustment
        return torch.clamp(density, min=0.01, max=1.0)
        
    def forward(self, x: torch.Tensor, recursion_level: int) -> torch.Tensor:
        """Compute D_k(x) for recursion level k"""
        if self.theoretical_mode:
            return self.compute_theoretical_density(x, recursion_level)
        
        # Learned density approximation
        batch_size = x.shape[0]
        level_tensor = torch.full((batch_size, 1), recursion_level, 
                                 dtype=x.dtype, device=x.device)
        input_with_level = torch.cat([x, level_tensor], dim=1)
        return self.density_net(input_with_level)


class RecursiveOperator(nn.Module):
    """
    Implementation of RSSN recursive operators (Triangle, Square, Circle, etc.)
    Each operator represents a different level of recursive complexity.
    
    Operators:
    - Triangle: Base recursion (x^x bounded)
    - Square: Nested recursion (Triangle^n(n))
    - Circle: Meta-recursion (Square^n(n))
    - Pentagon: Phase I recursion with fractal density
    - Hexagon: Phase II recursion
    """
    
    OPERATOR_TYPES = ['triangle', 'square', 'circle', 'pentagon', 'hexagon', 'aether']
    
    def __init__(self, input_dim: int, operator_type: str, max_recursion: int = 5,
                 enable_attention: bool = True):
        super().__init__()
        
        if operator_type not in self.OPERATOR_TYPES:
            raise ValueError(f"Unknown operator type: {operator_type}. "
                           f"Must be one of {self.OPERATOR_TYPES}")
        
        self.input_dim = input_dim
        self.operator_type = operator_type
        self.max_recursion = max_recursion
        self.enable_attention = enable_attention
        
        # Learnable parameters for recursive operations
        self.recursive_weights = nn.Parameter(
            torch.randn(max_recursion, input_dim, input_dim) * 0.1
        )
        self.scaling_factors = nn.Parameter(torch.ones(max_recursion))
        self.bias_terms = nn.Parameter(torch.zeros(max_recursion, input_dim))
        
        # Density function for this operator
        self.density_func = RecursiveDensityFunction(input_dim, max_recursion)
        
        # Attention mechanism for recursive steps
        if enable_attention:
            self.step_attention = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=4, batch_first=True
            )
        
        # Normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(max_recursion)
        ])
        
    def apply_triangle_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Triangle operator: bounded version of x^x"""
        W = self.recursive_weights[step]
        scale = self.scaling_factors[step]
        bias = self.bias_terms[step]
        
        # Bounded exponential operation
        base = torch.clamp(torch.abs(x), min=0.1, max=3.0)
        exponent = torch.clamp(x, max=2.0)
        
        # Differentiable approximation of x^x
        result = scale * torch.exp(exponent * torch.log(base + 1e-8))
        result = torch.clamp(result, max=1000.0)  # Prevent explosion
        
        # Apply linear transformation
        transformed = torch.matmul(result, W) + bias
        return self.layer_norms[step](transformed)
    
    def apply_square_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Square operator: nested Triangle application"""
        if step == 0:
            return self.apply_triangle_step(x, step)
        
        # Recursively apply triangle
        prev_result = self.apply_square_step(x, step - 1)
        return self.apply_triangle_step(prev_result, step)
    
    def apply_circle_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Circle operator: meta-recursive with density modulation"""
        density = self.density_func(x, step)
        
        if step == 0:
            return self.apply_square_step(x * density, step)
        
        # Meta-recursive application
        prev_result = self.apply_circle_step(x, step - 1)
        modulated_input = prev_result * density
        return self.apply_square_step(modulated_input, step)
    
    def apply_pentagon_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Pentagon operator: Circle^{D_3(n)}(n)"""
        # Calculate fractal density for fractional iteration
        density = self.density_func(x, 3)  # D_3(n)
        
        # Fractional iteration approximation
        iteration_count = torch.mean(density).item()
        
        result = x
        for _ in range(int(iteration_count) + 1):
            if torch.mean(self.density_func(result, step)) < 0.05:
                break
            result = self.apply_circle_step(result, step)
        
        return result
    
    def apply_hexagon_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Hexagon operator: Pentagon^{D_4(n)}(n)"""
        density = self.density_func(x, 4)  # D_4(n)
        
        # Even more complex fractional iteration
        iteration_count = torch.mean(density).item()
        
        result = x
        for _ in range(int(iteration_count * 2) + 1):
            if torch.mean(self.density_func(result, step)) < 0.02:
                break
            result = self.apply_pentagon_step(result, step)
        
        return result
    
    def apply_aether_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Aether operator: limit of all shape operators"""
        # Combine all previous operators with learned weights
        triangle_out = self.apply_triangle_step(x, step)
        square_out = self.apply_square_step(x, min(step, 2))
        circle_out = self.apply_circle_step(x, min(step, 1))
        
        # Learned combination
        combined = torch.stack([triangle_out, square_out, circle_out], dim=1)
        
        if self.enable_attention:
            attended, _ = self.step_attention(combined, combined, combined)
            result = torch.mean(attended, dim=1)
        else:
            result = torch.mean(combined, dim=1)
        
        return result
    
    def apply_recursive_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply one step of the specified recursive operation"""
        if self.operator_type == 'triangle':
            return self.apply_triangle_step(x, step)
        elif self.operator_type == 'square':
            return self.apply_square_step(x, step)
        elif self.operator_type == 'circle':
            return self.apply_circle_step(x, step)
        elif self.operator_type == 'pentagon':
            return self.apply_pentagon_step(x, step)
        elif self.operator_type == 'hexagon':
            return self.apply_hexagon_step(x, step)
        elif self.operator_type == 'aether':
            return self.apply_aether_step(x, step)
        else:
            # Fallback to simple linear transformation
            W = self.recursive_weights[step]
            return torch.matmul(x, W)
    
    def forward(self, x: torch.Tensor, adaptive_depth: bool = True) -> torch.Tensor:
        """Apply full recursive operation with optional adaptive depth"""
        current = x
        actual_depth = 0
        
        for step in range(self.max_recursion):
            density = self.density_func(current, step)
            mean_density = torch.mean(density)
            
            # Adaptive early stopping based on density
            if adaptive_depth and mean_density < 0.1:
                break
                
            current = self.apply_recursive_step(current, step)
            actual_depth = step + 1
            
            # Safety check for numerical stability
            if torch.isnan(current).any() or torch.isinf(current).any():
                warnings.warn(f"Numerical instability in {self.operator_type} at step {step}")
                break
        
        return current


class RecursiveLayer(nn.Module):
    """
    A neural network layer that combines multiple recursive operators with attention
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 operators: List[str] = ['triangle', 'square', 'circle'],
                 max_recursion: int = 3,
                 enable_residual: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.operators = operators
        self.enable_residual = enable_residual
        
        # Validate operators
        invalid_ops = set(operators) - set(RecursiveOperator.OPERATOR_TYPES)
        if invalid_ops:
            raise ValueError(f"Invalid operators: {invalid_ops}")
        
        # Create recursive operators
        self.recursive_ops = nn.ModuleList([
            RecursiveOperator(input_dim, op_type, max_recursion) 
            for op_type in operators
        ])
        
        # Operator attention mechanism
        self.operator_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=min(4, input_dim // 16), batch_first=True
        )
        
        # Combination and projection networks
        combined_dim = len(operators) * input_dim
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, output_dim)
        )
        
        # Residual connection
        if enable_residual and input_dim == output_dim:
            self.residual_projection = nn.Identity()
        elif enable_residual:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through recursive layer"""
        # Apply each recursive operator
        operator_outputs = []
        for op in self.recursive_ops:
            op_output = op(x)
            operator_outputs.append(op_output)
        
        # Stack outputs for attention
        stacked_outputs = torch.stack(operator_outputs, dim=1)  # [batch, num_ops, features]
        
        # Apply attention to weight operators
        attended_output, attention_weights = self.operator_attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )
        
        # Combine attended outputs
        flattened = attended_output.flatten(start_dim=1)
        output = self.combiner(flattened)
        
        # Residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(x)
            output = output + residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        if return_attention:
            return output, attention_weights
        return output


class RGNN(nn.Module):
    """
    Complete Recursive Graded Neural Network
    
    This is the main RGNN model that implements the theoretical framework
    from the RGNN paper using recursive operators and fractal density functions.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int,
                 operators: List[str] = ['triangle', 'square', 'circle'],
                 max_recursion: int = 3,
                 enable_residual: bool = True,
                 dropout: float = 0.1,
                 theoretical_mode: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.operators = operators
        self.theoretical_mode = theoretical_mode
        
        # Input projection with normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Recursive layers
        self.recursive_layers = nn.ModuleList()
        layer_dims = [hidden_dims[0]] + hidden_dims[1:] + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            layer = RecursiveLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                operators=operators,
                max_recursion=max_recursion,
                enable_residual=enable_residual
            )
            self.recursive_layers.append(layer)
        
        # Global recursive signature computation
        self.signature_computer = RecursiveDensityFunction(
            hidden_dims[-1] if hidden_dims else input_dim
        )
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Enable theoretical mode if requested
        if theoretical_mode:
            self.enable_theoretical_mode()
    
    def enable_theoretical_mode(self):
        """Enable theoretical RSSN calculations across all components"""
        self.theoretical_mode = True
        self.signature_computer.enable_theoretical_mode()
        for layer in self.recursive_layers:
            for op in layer.recursive_ops:
                op.density_func.enable_theoretical_mode()
    
    def disable_theoretical_mode(self):
        """Use learned approximations"""
        self.theoretical_mode = False
        self.signature_computer.disable_theoretical_mode()
        for layer in self.recursive_layers:
            for op in layer.recursive_ops:
                op.density_func.disable_theoretical_mode()
        
    def forward(self, x: torch.Tensor, 
                return_signatures: bool = True,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through RGNN"""
        
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[1]}")
        
        # Project input
        h = self.input_projection(x)
        
        # Track recursive signatures and attention weights
        signatures = []
        attention_weights = []
        
        # Apply recursive layers
        for i, layer in enumerate(self.recursive_layers):
            if return_attention:
                h, attn = layer(h, return_attention=True)
                attention_weights.append(attn)
            else:
                h = layer(h)
            
            # Compute recursive signature for this layer
            if return_signatures:
                signature = self.signature_computer(h, recursion_level=i)
                signatures.append(signature)
        
        # Final output projection
        output = self.output_projection(h)
        
        # Prepare return dictionary
        result = {'output': output, 'hidden_state': h}
        
        if return_signatures:
            result['recursive_signatures'] = signatures
            # Compute final recursive complexity
            final_signature = self.signature_computer(h, recursion_level=len(self.recursive_layers))
            result['final_signature'] = final_signature
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result
    
    def compute_recursive_complexity(self) -> float:
        """Compute the overall recursive complexity of the network"""
        total_complexity = 0.0
        
        for layer in self.recursive_layers:
            for op in layer.recursive_ops:
                # Sum scaling factors as a measure of complexity
                complexity = torch.sum(torch.abs(op.scaling_factors)).item()
                total_complexity += complexity
        
        return total_complexity
    
    def get_recursive_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the recursive structure"""
        stats = {
            'total_layers': len(self.recursive_layers),
            'operators_per_layer': len(self.operators),
            'total_operators': len(self.recursive_layers) * len(self.operators),
            'recursive_complexity': self.compute_recursive_complexity(),
            'theoretical_mode': self.theoretical_mode,
            'parameter_count': sum(p.numel() for p in self.parameters())
        }
        
        # Per-layer statistics
        layer_stats = []
        for i, layer in enumerate(self.recursive_layers):
            layer_info = {
                'layer_index': i,
                'input_dim': layer.input_dim,
                'output_dim': layer.output_dim,
                'operators': layer.operators,
                'parameter_count': sum(p.numel() for p in layer.parameters())
            }
            layer_stats.append(layer_info)
        
        stats['layer_statistics'] = layer_stats
        return stats


class RecursiveStructureLoss(nn.Module):
    """
    Specialized loss function for RGNN that incorporates recursive structure regularization
    """
    
    def __init__(self, 
                 task_weight: float = 1.0, 
                 structure_weight: float = 0.1, 
                 smoothness_weight: float = 0.05,
                 diversity_weight: float = 0.02,
                 complexity_weight: float = 0.01):
        super().__init__()
        
        self.task_weight = task_weight
        self.structure_weight = structure_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
        self.complexity_weight = complexity_weight
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor,
                task_type: str = 'regression') -> Dict[str, torch.Tensor]:
        
        predictions = outputs['output']
        
        # Main task loss
        if task_type == 'classification':
            task_loss = F.cross_entropy(predictions, targets.long())
        else:  # regression
            task_loss = F.mse_loss(predictions, targets)
        
        total_loss = self.task_weight * task_loss
        loss_components = {'task_loss': task_loss}
        
        # Structural regularization terms
        if 'recursive_signatures' in outputs and outputs['recursive_signatures']:
            signatures = outputs['recursive_signatures']
            signature_tensor = torch.stack(signatures, dim=1)  # [batch, layers, 1]
            
            # Diversity loss: signatures should be different across layers
            if signature_tensor.shape[1] > 1:
                signature_var = torch.var(signature_tensor, dim=1)
                diversity_loss = -torch.mean(signature_var)  # Negative because we want high variance
                total_loss += self.diversity_weight * diversity_loss
                loss_components['diversity_loss'] = diversity_loss
            
            # Smoothness loss: adjacent signatures shouldn't change too rapidly
            if signature_tensor.shape[1] > 1:
                signature_diff = torch.diff(signature_tensor, dim=1)
                smoothness_loss = torch.mean(signature_diff ** 2)
                total_loss += self.smoothness_weight * smoothness_loss
                loss_components['smoothness_loss'] = smoothness_loss
            
            # Structure loss: encourage meaningful density values
            mean_signature = torch.mean(signature_tensor)
            structure_loss = torch.abs(mean_signature - 0.5)  # Encourage mid-range densities
            total_loss += self.structure_weight * structure_loss
            loss_components['structure_loss'] = structure_loss
        
        loss_components['total_loss'] = total_loss
        return loss_components


# Factory functions for easy model creation
def create_rgnn_for_task(task_type: str = 'classification', 
                        input_dim: int = None,
                        output_dim: int = None,
                        complexity: str = 'medium',
                        theoretical_mode: bool = False,
                        **kwargs) -> RGNN:
    """
    Factory function to create RGNN models for different tasks
    
    Args:
        task_type: 'classification', 'regression', 'symbolic_reasoning'
        input_dim: Input dimension (required)
        output_dim: Output dimension (required)
        complexity: 'simple', 'medium', 'complex'
        theoretical_mode: Use theoretical RSSN calculations
        **kwargs: Additional model parameters
    
    Returns:
        Configured RGNN model
    """
    
    if input_dim is None or output_dim is None:
        raise ValueError("input_dim and output_dim are required")
    
    # Default configurations based on complexity
    configs = {
        'simple': {
            'hidden_dims': [64, 32],
            'operators': ['triangle', 'square'],
            'max_recursion': 2
        },
        'medium': {
            'hidden_dims': [128, 64, 32],
            'operators': ['triangle', 'square', 'circle'],
            'max_recursion': 3
        },
        'complex': {
            'hidden_dims': [256, 128, 64, 32],
            'operators': ['triangle', 'square', 'circle', 'pentagon'],
            'max_recursion': 5
        }
    }
    
    config = configs.get(complexity, configs['medium'])
    config.update(kwargs)  # Override with user parameters
    
    # Task-specific adjustments
    if task_type == 'symbolic_reasoning':
        config['operators'] = ['triangle', 'square', 'circle', 'pentagon']
        config['max_recursion'] = max(config['max_recursion'], 4)
    
    return RGNN(
        input_dim=input_dim,
        output_dim=output_dim,
        theoretical_mode=theoretical_mode,
        **config
    )


def train_rgnn(model: RGNN, 
               train_loader, 
               val_loader=None,
               epochs: int = 100,
               lr: float = 1e-3,
               task_type: str = 'regression',
               device: str = 'cpu',
               verbose: bool = True) -> Dict[str, List[float]]:
    """
    Training loop for RGNN with recursive structure monitoring
    
    Args:
        model: RGNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        lr: Learning rate
        task_type: 'classification' or 'regression'
        device: Device to train on
        verbose: Print training progress
    
    Returns:
        Dictionary containing training history
    """
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
    )
    
    loss_fn = RecursiveStructureLoss()
    
    history = {
        'train_task_loss': [],
        'train_total_loss': [],
        'val_task_loss': [],
        'val_total_loss': [],
        'recursive_complexity': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = {'task': 0, 'total': 0}
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss_dict = loss_fn(outputs, batch_y, task_type=task_type)
            
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses['task'] += loss_dict['task_loss'].item()
            train_losses['total'] += loss_dict['total_loss'].item()
        
        # Average training losses
        avg_train_task_loss = train_losses['task'] / len(train_loader)
        avg_train_total_loss = train_losses['total'] / len(train_loader)
        
        history['train_task_loss'].append(avg_train_task_loss)
        history['train_total_loss'].append(avg_train_total_loss)
        
        # Validation phase
        val_task_loss = val_total_loss = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    outputs = model(batch_x)
                    loss_dict = loss_fn(outputs, batch_y, task_type=task_type)
                    
                    val_task_loss += loss_dict['task_loss'].item()
                    val_total_loss += loss_dict['total_loss'].item()
            
            val_task_loss /= len(val_loader)
            val_total_loss /= len(val_loader)
            
            history['val_task_loss'].append(val_task_loss)
            history['val_total_loss'].append(val_total_loss)
            
            # Learning rate scheduling
            scheduler.step(val_task_loss)
            
            # Early stopping
            if val_task_loss < best_val_loss:
                best_val_loss = val_task_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Record complexity and learning rate
        complexity = model.compute_recursive_complexity()
        history['recursive_complexity'].append(complexity)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Verbose output
        if verbose and epoch % 10 == 0:
            if val_loader is not None:
                print(f"Epoch {epoch:3d}: Train Loss = {avg_train_task_loss:.4f}, "
                      f"Val Loss = {val_task_loss:.4f}, "
                      f"Complexity = {complexity:.4f}, "
                      f"LR = {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss = {avg_train_task_loss:.4f}, "
                      f"Complexity = {complexity:.4f}")
    
    return history


# Utility functions
def analyze_recursive_structure(model: RGNN, 
                               sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze the recursive structure of a trained RGNN
    
    Args:
        model: Trained RGNN model
        sample_input: Sample input to analyze
    
    Returns:
        Analysis results
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(sample_input, return_signatures=True, return_attention=True)
    
    analysis = {
        'model_statistics': model.get_recursive_statistics(),
        'output_shape': outputs['output'].shape,
        'signature_evolution': [sig.mean().item() for sig in outputs['recursive_signatures']],
        'final_complexity': outputs['final_signature'].mean().item()
    }
    
    if 'attention_weights' in outputs:
        # Analyze attention patterns
        attention_stats = []
        for i, attn in enumerate(outputs['attention_weights']):
            attention_stats.append({
                'layer': i,
                'mean_attention': attn.mean().item(),
                'attention_entropy': -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean().item()
            })
        analysis['attention_statistics'] = attention_stats
    
    return analysis


def save_rgnn_checkpoint(model: RGNN, 
                        optimizer: torch.optim.Optimizer,
                        history: Dict[str, List[float]],
                        filepath: str):
    """Save RGNN model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'output_dim': model.output_dim,
            'operators': model.operators,
            'theoretical_mode': model.theoretical_mode
        },
        'training_history': history,
        'model_statistics': model.get_recursive_statistics()
    }
    torch.save(checkpoint, filepath)


def load_rgnn_checkpoint(filepath: str, device: str = 'cpu') -> Tuple[RGNN, Dict]:
    """Load RGNN model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model
    config = checkpoint['model_config']
    model = RGNN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint


# Export main classes and functions
__all__ = [
    'RGNN',
    'RecursiveLayer', 
    'RecursiveOperator',
    'RecursiveDensityFunction',
    'RecursiveStructureLoss',
    'create_rgnn_for_task',
    'train_rgnn',
    'analyze_recursive_structure',
    'save_rgnn_checkpoint',
    'load_rgnn_checkpoint'
]
