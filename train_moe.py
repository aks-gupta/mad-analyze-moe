import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer


# -----------------------------
# Data
# -----------------------------
class JsonlSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
        self.items = []
        self.tok = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"].strip()
                output = obj["output"].strip()

                # build causal LM training text
                # You can change <assistant> token to something else if you like
                text = f"{prompt}\n<assistant>\n{output}"
                ids = self.tok(
                    text,
                    truncation=True,
                    max_length=max_len,
                    return_tensors=None,
                    add_special_tokens=True,
                )["input_ids"]
                self.items.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids = self.items[idx]
        return ids


def pad_collate(batch, pad_id):
    maxlen = max(x.size(0) for x in batch)
    out = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
        attn[i, : x.size(0)] = 1
    labels = out.clone()
    labels[~attn] = -100  # ignore padding in loss
    return out, attn, labels


# -----------------------------
# MoE Blocks
# -----------------------------
class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # SwiGLU: project to 2*d_ff
        self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class Top2Router(nn.Module):
    def __init__(self, d_model, n_experts, noise_std=1e-2):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False)
        self.noise_std = noise_std
        self.n_experts = n_experts

    def forward(self, x, train=True):
        # x: [B, T, H]
        logits = self.proj(x)
        if train and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gate = F.softmax(logits, dim=-1)  # [B, T, E]
        top2_val, top2_idx = torch.topk(gate, k=2, dim=-1)  # [B,T,2]
        return top2_val, top2_idx, gate


class MoEFFN(nn.Module):
    """
    Naive reference MoE: Top-2 with simple dispatch.
    For speed at scale: use fused kernels (DeepSpeed-MoE / Tutel).
    """
    def __init__(self, d_model, d_ff, n_experts=8, capacity_factor=1.25):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.router = Top2Router(d_model, n_experts)

    def forward(self, x, train=True):
        B, T, H = x.shape
        top2_val, top2_idx, gate = self.router(x, train=train)
        tokens = B * T
        cap = max(1, int(self.capacity_factor * (tokens * 2 / self.n_experts)))

        y = torch.zeros_like(x)
        # load balancing stats
        prob_per_expert = gate.mean(dim=(0, 1))
        used_per_expert = torch.zeros(self.n_experts, device=x.device)

        flat_x = x.reshape(-1, H)
        flat_y = y.reshape(-1, H)

        for k in range(2):
            idx = top2_idx[..., k].reshape(-1)        # [B*T]
            val = top2_val[..., k].reshape(-1, 1)     # [B*T,1]

            for e in range(self.n_experts):
                mask = (idx == e)
                if not mask.any():
                    continue
                sel = mask.nonzero(as_tuple=False).squeeze(-1)
                # capacity clipping (naive)
                take = min(sel.numel(), cap - int(used_per_expert[e].item()))
                if take <= 0:
                    continue
                sel = sel[:take]
                out = self.experts[e](flat_x.index_select(0, sel))
                # weighted add
                weighted = out * val.index_select(0, sel)
                # scatter-add
                flat_y.index_copy_(0, sel, flat_y.index_select(0, sel) + weighted)
                used_per_expert[e] += take

        y = flat_y.view(B, T, H)

        # Switch-style load balancing aux
        me = 1e-9
        frac_used = used_per_expert.clamp_min(me) / (tokens * 2 + me)
        balance_loss = (prob_per_expert * frac_used).sum() * self.n_experts

        return y, balance_loss

class DenseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)   # 512 → 2048
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)   # 2048 → 512
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -----------------------------
# Tiny Transformer with MoE in FFN
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        qkv = self.qkv(x)  # [B, T, 3*H]
        qkv = qkv.view(B, T, 3, self.h, self.dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, heads, dk]

        # move heads into batch dim
        q = q.permute(0, 2, 1, 3)  # [B, heads, T, dk]
        k = k.permute(0, 2, 1, 3)  # [B, heads, T, dk]
        v = v.permute(0, 2, 1, 3)  # [B, heads, T, dk]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,heads,T,T]

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(causal_mask == 0, -1e9)

        att = F.softmax(att, dim=-1)
        out = att @ v  # [B,heads,T,dk]

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H)  # restore [B,T,H]
        return self.proj(out)



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_experts=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = n_experts is not None
        if self.use_moe:
            self.ff = MoEFFN(d_model, d_ff, n_experts=n_experts)
        else:
            self.ff_dense = DenseFFN(d_model, d_ff)

    def forward(self, x, train=True):
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            h, bal = self.ff(self.ln2(x), train=train)
            x = x + h
            return x, bal
        else:
            x = x + self.ff_dense(self.ln2(x))
            return x, torch.tensor(0.0, device=x.device)


class TinyMoELM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6, d_ff=1536, n_experts=8):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        blocks = []
        for i in range(n_layers):
            # put MoE in every other block
            use_moe = (i % 2 == 1)
            blocks.append(TransformerBlock(d_model, n_heads, d_ff, n_experts if use_moe else None))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, train=True):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        balance_losses = 0.0
        for blk in self.blocks:
            x, bal = blk(x, train=train)
            balance_losses = balance_losses + bal
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, balance_losses


# -----------------------------
# Train
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_moe.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = JsonlSFTDataset(args.data, tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id))

    model = TinyMoELM(vocab_size=tokenizer.vocab_size,
                      d_model=512, n_layers=8, n_heads=8, d_ff=2048, n_experts=8).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    for epoch in range(args.epochs):
        model.train()
        total, total_aux = 0.0, 0.0
        for step, (inp, attn, labels) in enumerate(dl):
            inp = inp.to(args.device)
            labels = labels.to(args.device)

            logits, bal_loss = model(inp, train=True)
            # standard causal LM loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_aux += float(bal_loss.item())
            # add small load balancing loss
            loss = loss + 0.03 * bal_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {total/(step+1):.4f} aux {total_aux/(step+1):.4f}")

        # save tiny checkpoint
        ckpt = f"tiny_moe_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "tokenizer": tokenizer.name_or_path}, ckpt)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
