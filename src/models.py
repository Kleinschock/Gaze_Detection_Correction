import torch
import torch.nn as nn


class FFNN(nn.Module):
    """
    A flexible Feed-Forward Neural Network (FFNN) model that can have multiple
    hidden layers with varying sizes.
    """
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list = [256, 128], dropout: float = 0.5):
        super().__init__()
        
        layers = [nn.Flatten()]
        current_input_size = input_size
        
        # Create a stack of hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_input_size = hidden_size
            
        # Add the final output layer
        layers.append(nn.Linear(current_input_size, num_classes))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GRUNet(nn.Module):
    """
    A Gated Recurrent Unit (GRU) network designed for sequence modeling.
    This model can have multiple layers with varying hidden sizes, allowing for
    more complex feature extraction hierarchies.
    """
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list = [128, 64], dropout: float = 0.5):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        # Create a stack of GRU layers
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.GRU(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,  # Each element in ModuleList is a single-layer GRU
                batch_first=True
            ))
            current_input_size = hidden_size # The input to the next layer is the output of this one

        # Add a dropout layer for regularization before the final classification layer.
        self.dropout = nn.Dropout(dropout)
        # The final fully connected layer takes the output of the last GRU layer.
        self.fc = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        out = x
        # Pass the input through each GRU layer in the stack
        for layer in self.layers:
            out, _ = layer(out)
            # We can apply dropout between layers if desired, but for now we apply at the end
        
        # We take the output from the last time step of the final layer.
        last_time_step_out = out[:, -1, :]
        
        # Apply dropout before the final layer.
        last_time_step_out = self.dropout(last_time_step_out)
        
        # Pass it through the fully connected layer.
        return self.fc(last_time_step_out)


class SpotterNet(nn.Module):
    """
    A lightweight Feed-Forward Neural Network designed for the binary task of
    spotting motion vs. no motion on a per-frame basis.
    """
    def __init__(self, input_size: int, hidden_sizes: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_input_size = hidden_size
            
        # Output layer for binary classification (1 output neuron)
        layers.append(nn.Linear(current_input_size, 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is a single frame, already flattened.
        return self.layers(x)


def get_model(name: str, input_size: int, num_classes: int, model_params: dict = None) -> nn.Module:
    """
    Factory function to instantiate a model by name with specified parameters.
    This allows for dynamic creation of models during hyperparameter tuning.
    """
    model_params = model_params or {}
    
    if name.lower() == 'ffnn':
        # Rename 'dropout_rate' to 'dropout' to match the model's constructor argument
        if 'dropout_rate' in model_params:
            model_params['dropout'] = model_params.pop('dropout_rate')
        print(f"Creating FFNN model with params: {model_params}")
        return FFNN(input_size, num_classes, **model_params)
    elif name.lower() == 'gru':
        # Rename 'dropout_rate' to 'dropout' to match the model's constructor argument
        if 'dropout_rate' in model_params:
            model_params['dropout'] = model_params.pop('dropout_rate')
        print(f"Creating GRU model with params: {model_params}")
        return GRUNet(input_size, num_classes, **model_params)
    elif name.lower() == 'spotter':
        if 'dropout_rate' in model_params:
            model_params['dropout'] = model_params.pop('dropout_rate')
        print(f"Creating SpotterNet model with params: {model_params}")
        # num_classes is not needed for spotter, it's always 1
        return SpotterNet(input_size, **model_params)
    else:
        raise ValueError(f"Unknown model architecture: '{name}'. Choose 'ffnn', 'gru', or 'spotter'.")
