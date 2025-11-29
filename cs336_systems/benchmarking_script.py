# workflow
# Train + Record
# Record Tool --> timeit module
# Record Problem --> CUDA Calls are asynchronous
from dataclasses import dataclass, asdict
import torch.nn as nn
import numpy as np
import torch
from cs336_basics.model import * # type: ignore
from cs336_basics.data import * # type: ignore
import timeit

data_conf = {
    "data_path": "../data/valid.bin",
    "batch_size": 4,
    "context_length": 256,
    "step": 5,
}

@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0

def forward_pass(model, input, device):
    logits = model(input)

    if device == "cuda":
        torch.cuda.synchronize()

    return logits

def benchmarking():
    # initial
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")
    config = TransformerConfig()
    model = BasicsTransformerLM(**asdict(config)) # type: ignore
    model.to(device)
    # data
    data = np.memmap(data_conf["data_path"], dtype="uint16", mode="r")
  
    # train
    print("Warming up...")
    for _ in range(5):
        X, Y = get_batch(data, data_conf["batch_size"], data_conf["context_length"], device) # type: ignore
        forward_pass(model, X, device)

    X, Y = get_batch(data, data_conf["batch_size"], data_conf["context_length"], device) # type: ignore

    stmt_code = lambda: forward_pass(model, X, device)

    total_time = timeit.timeit(stmt_code, number=100)
    avg_time = total_time / 100
    print(f"Average Forward Pass Time: {avg_time * 1000:.2f} ms")


    

if __name__ == "__main__":
    benchmarking()