import torch
import torch.nn as nn


class TcPredictor(nn.Module):
    """
    Advanced Deep Residual Network for superconductor Tc prediction.
    Using deeper blocks, bottlenecks, and improved regularization.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 256,
            dropout_rate: float = 0.2,
            num_blocks: int = 4,
            expansion: int = 2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                expansion=expansion
            )
            for _ in range(num_blocks)
        ])

        # Feature pooling / skip connection from input projection to head
        self.skip_all = nn.Linear(hidden_size, hidden_size)

        # Output head
        self.output_head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_proj(x)
        
        identity = self.skip_all(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = x + identity # Global residual connection
        return self.output_head(x)


class ResidualBlock(nn.Module):
    """
    Bottleneck Residual Block: Linear -> Batchnorm -> SiLU -> Linear -> Batchnorm -> SiLU -> Linear -> Dropout
    """
    def __init__(self, hidden_size: int, dropout_rate: float, expansion: int = 2):
        super().__init__()
        
        mid_size = hidden_size * expansion
        
        self.layers = nn.Sequential(
            # Bottleneck expansion
            nn.Linear(hidden_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.SiLU(),
            
            # Feature transformation
            nn.Linear(mid_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            # Compression
            nn.Linear(mid_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        
        self.gate = nn.SiLU()

    def forward(self, x):
        return self.gate(x + self.layers(x))