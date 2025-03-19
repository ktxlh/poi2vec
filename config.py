import torch

# Parameters
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ltype = torch.int64 if not torch.cuda.is_available() else torch.cuda.LongTensor
ftype = torch.float32 if not torch.cuda.is_available() else torch.cuda.FloatTensor

# Model Hyperparameters
feat_dim = 200
route_depth = 16
route_count = 4
context_len = 32

# Training Parameters
batch_size = 128
num_epochs = 30
evaluate_every = 1
patience = 5
