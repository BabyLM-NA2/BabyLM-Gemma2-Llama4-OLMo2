import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class RWKVTimeMix(nn.Module):
    def __init__(self, dim, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        
        # Learnable parameters
        self.time_decay = nn.Parameter(torch.empty(dim))
        self.time_curve = nn.Parameter(torch.tensor([-(i+1) for i in range(dim)]).float())
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        # Projections
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        # Initialization
        with torch.no_grad():
            ratio_0_to_1 = (layer_id / 6)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / 12)  # 1 to ~0
            
            # Time decay
            decay_speed = torch.ones(dim)
            for h in range(dim):
                decay_speed[h] = -5 + 8 * (h / (dim-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay.data = decay_speed
            
            # Time mixing
            x = torch.ones(1, 1, dim)
            for h in range(dim):
                x[0, 0, h] = h / dim
            self.time_curve.data = x[0, 0]

    def forward(self, x, state):
        # Token shift and mixing
        xx = self.time_shift(x)
        xk = x * self.time_curve + xx * (1 - self.time_curve)
        xv = x * self.time_curve + xx * (1 - self.time_curve)
        xr = x * self.time_curve + xx * (1 - self.time_curve)
        
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        
        # Time decay and state management
        time_decay = self.time_decay.float()
        time_decay = -torch.exp(time_decay)
        
        # State update equation
        if state is None:
            state = torch.zeros_like(k)
        
        new_state = k * v + torch.exp(time_decay) * state
        output = sr * new_state
        
        return self.output(output), new_state

class RWKVChannelMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        # Projections
        self.key = nn.Linear(dim, 4*dim, bias=False)
        self.value = nn.Linear(4*dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        
        # Initialization
        with torch.no_grad():
            # ratio_1_to_almost0 = 1.0 - (layer_id / 12)  # From paper
            
            # Time mixing
            x = torch.ones(1, 1, dim)
            for h in range(dim):
                x[0, 0, h] = h / dim
            self.time_curve = nn.Parameter(x)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_curve + xx * (1 - self.time_curve)
        xr = x * self.time_curve + xx * (1 - self.time_curve)
        
        k = self.key(xk)
        k = torch.square(torch.relu(k))  # GeGLU-like activation
        v = self.value(k)
        sr = torch.sigmoid(self.receptance(xr))
        
        return sr * v

class RWKVBlock(nn.Module):
    def __init__(self, dim, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.timemix = RWKVTimeMix(dim, layer_id)
        self.ln2 = nn.LayerNorm(dim)
        self.channelmix = RWKVChannelMix(dim)
        
    def forward(self, x, state):
        # Time mixing branch
        residual = x
        x = self.ln1(x)
        tm, new_state = self.timemix(x, state)
        x = residual + tm
        
        # Channel mixing branch
        residual = x
        x = self.ln2(x)
        x = residual + self.channelmix(x)
        
        return x, new_state

class RWKV(nn.Module):
    def __init__(self, vocab_size, dim, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, layer_id=i) for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.emb.weight
        
        # Initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Orthogonal initialization for receptance weights
            if 'receptance' in str(module):
                torch.nn.init.orthogonal_(module.weight, gain=0.1)
            else:
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, tokens, states=None):
        x = self.emb(tokens)
        new_states = []
        
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, states[i] if states else None)
            new_states.append(new_state)
            
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states

# Usage Example
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 50277  # Typical RWKV tokenizer size
    dim = 768           # Model dimension
    n_layers = 12       # Number of RWKV blocks
    
    model = RWKV(vocab_size, dim, n_layers)
    
    # Sample input
    tokens = torch.randint(0, vocab_size, (1, 1024))  # (batch, seq_len)
    
    # Initial state (None for first pass)
    states = None
    
    # Forward pass
    logits, new_states = model(tokens, states)
    
    print("Output shape:", logits.shape)  # Should be (1, 1024, vocab_size)
