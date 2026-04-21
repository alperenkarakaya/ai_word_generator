"""
Transformer eğitim döngüsü.

Kullanım (Colab/Kaggle):
  python transformer/train.py --data_dir data --tokenizer tokenizer/spm.model \
      --steps 20000 --batch_size 32 --block_size 256

Çıktılar:
  checkpoints/transformer_step_*.pt
  checkpoints/transformer.pt    (en iyi val loss'a sahip)
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F  # noqa: F401  (AMP testleri için)
# PyTorch 2.1: torch.cuda.amp.GradScaler/autocast. 2.4+ torch.amp.* tercih edilir.
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import load_tokenizer
from transformer.dataset import BinDataset
from transformer.model import GPT, GPTConfig


def build_lr_lambda(warmup_steps: int, max_steps: int, min_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    use_amp = str(device).startswith("cuda")
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        losses = []
        for _ in range(eval_iters):
            x, y = ds.get_batch(batch_size, device=device)
            with autocast(enabled=use_amp, dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tokenizer", default="tokenizer/spm.model")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--resume", default=None, help="Devam edilecek checkpoint yolu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = load_tokenizer(args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Model params: {model.num_parameters() / 1e6:.2f}M")

    train_ds = BinDataset(os.path.join(args.data_dir, "train.bin"), cfg.block_size)
    val_ds = BinDataset(os.path.join(args.data_dir, "val.bin"), cfg.block_size)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    lr_lambda = build_lr_lambda(args.warmup_steps, args.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = GradScaler(enabled=device == "cuda")

    start_step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Devam ediliyor: step {start_step}, best_val {best_val:.4f}")

    model.train()
    use_amp = device == "cuda"
    t0 = time.time()
    for step in range(start_step, args.steps):
        x, y = train_ds.get_batch(args.batch_size, device=device)
        with autocast(enabled=use_amp, dtype=torch.float16):
            _, loss, _ = model(x, targets=y)

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        if step % 50 == 0:
            elapsed = time.time() - t0
            tok_per_s = (step - start_step + 1) * args.batch_size * args.block_size / max(elapsed, 1e-6)
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} "
                  f"| tok/s {tok_per_s:.0f}")

        if (step + 1) % args.eval_interval == 0:
            losses = estimate_loss(model, train_ds, val_ds, args.batch_size, device, args.eval_iters)
            print(f"  >>> step {step+1} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(model, optim, scheduler, cfg, step + 1, best_val,
                                os.path.join(args.out_dir, "transformer.pt"))
                print(f"  >>> en iyi val. Kaydedildi: transformer.pt")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optim, scheduler, cfg, step + 1, best_val,
                            os.path.join(args.out_dir, f"transformer_step_{step+1}.pt"))

    print("Eğitim tamamlandı.")


def save_checkpoint(model, optim, scheduler, cfg, step, best_val, path):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg.__dict__,
        "step": step,
        "best_val": best_val,
    }, path)


if __name__ == "__main__":
    main()
