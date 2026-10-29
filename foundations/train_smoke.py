"""Quick synthetic-data smoke training; it checks gradients, not real-world accuracy.
Run: python -m foundations.train_smoke --steps 10
"""
import argparse
import torch
from torch import nn
from .mini_models import MiniViT, MiniAudioTransformer


def train_one(model, inputs, labels, steps, name):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(inputs), labels)
        loss.backward()
        opt.step()
        if first is None: first = loss.item()
        last = loss.item()
    print(f"{name}: ran {steps} steps; first_loss={first:.4f}; last_loss={last:.4f}; synthetic data only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(7)
    train_one(MiniViT(), torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)), args.steps, 'MiniViT')
    train_one(MiniAudioTransformer(), torch.randn(2, 16000), torch.randint(0, 10, (2,)), args.steps, 'MiniAudioTransformer')

if __name__ == '__main__': main()
