"""MAE (Masked Autoencoder) 域内自监督预训练 —— 工单 #48。

在 63,100 张主办方眼底图（train+valid+test 的 cache 图片，**不用标签**）上做
掩码自编码预训练，得到「眼底域」骨干，再拿去微调二分类，缩小 valid→test 泛化差。

编码器：vit_small_patch16_224（timm），解码器：轻量 Transformer（embed 192 / depth 4）。
掩码率 75%，MSE 重建 masked patch 像素。完全合规（只用主办方图片，无外部数据）。

用法：
  python pretrain_mae.py --cache-root /root/autodl-tmp/glaucoma_first/cache768 \
      --epochs 40 --batch-size 256 --lr 1.5e-4 --out mae_fundus.pt

输出 mae_fundus.pt：{"model": encoder_state_dict, "arch": "vit_small_patch16_224",
"img_size": 224, "mask_ratio": 0.75, ...}，encoder 键与 timm vit_small 对齐，
可直接用 timm.create_model("vit_small_patch16_224").load_state_dict(sd, strict=False) 微调。
"""
import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 数据：所有 split 的 cache JPEG，无标签
# --------------------------------------------------------------------------- #
class FundusSSL(Dataset):
    def __init__(self, cache_root, size=224):
        self.size = size
        files = []
        for split in ("train", "valid", "test"):
            files += sorted(glob.glob(os.path.join(cache_root, split, "*.jpg")))
        self.files = files
        print(f"[SSL] {len(files)} images from {cache_root}", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = img.resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        std = torch.tensor(IMG_STD).view(3, 1, 1)
        return (x - mean) / std


# --------------------------------------------------------------------------- #
# MAE 模型（encoder 复用 timm vit_small）
# --------------------------------------------------------------------------- #
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=True):
    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(gw, gh)  # w, h
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    pos = get_1d_sincos_from_grid(embed_dim, grid)
    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim]), pos], axis=0)
    return torch.from_numpy(pos).float()


def get_1d_sincos_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos(embed_dim, positions):
    if embed_dim % 2 != 0:
        embed_dim += 1
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = positions.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def random_masking(x, mask_ratio):
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, enc_embed=384, enc_depth=12,
                 enc_heads=6, dec_embed=192, dec_depth=4, dec_heads=3,
                 mask_ratio=0.75, norm_pix_loss=False):
        super().__init__()
        import timm
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 编码器：timm vit_small（patch_embed + blocks + norm + cls/pos）
        vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed          # 可学习 [1, 197, 384]
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.enc_embed = enc_embed

        # 解码器
        self.decoder_embed = nn.Linear(enc_embed, dec_embed)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed))
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(dec_embed, img_size // patch_size, img_size // patch_size),
            requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            _Block(dec_embed, dec_heads) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_embed)
        self.decoder_pred = nn.Linear(dec_embed, patch_size ** 2 * 3)

        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * 3)

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)                       # [N, L, D]
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio)
        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=None):
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class _Block(nn.Module):
    """轻量 decoder Transformer block（pre-norm + MLP）。"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# 训练主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True, help="cache768 目录（含 train/valid/test）")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="mae_fundus.pt")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FundusSSL(args.cache_root, size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=(args.workers > 0))

    model = MaskedAutoencoder(img_size=args.img_size, mask_ratio=args.mask_ratio).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MAE] params={n_params/1e6:.1f}M  mask={args.mask_ratio}  imgs={len(ds)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    iters_per_epoch = len(dl)
    total_iters = iters_per_epoch * args.epochs
    warmup_iters = iters_per_epoch * args.warmup_epochs

    def lr_at(it):
        if it < warmup_iters:
            return args.lr * it / max(1, warmup_iters)
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    model.train()
    it = 0
    t0 = time.time()
    for ep in range(args.epochs):
        ep_loss = 0.0
        for imgs in dl:
            lr = lr_at(it)
            for g in opt.param_groups:
                g["lr"] = lr
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss, _, _ = model(imgs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            it += 1
            if it % args.log_every == 0:
                el = time.time() - t0
                print(f"  ep{ep+1} it{it}/{total_iters} loss={ep_loss/args.log_every:.4f} "
                      f"lr={lr:.2e} {el/60:.1f}min", flush=True)
                ep_loss = 0.0
        print(f"=== epoch {ep+1}/{args.epochs} done ===", flush=True)
        if (ep + 1) % 10 == 0:
            ckpt = {"model": model.state_dict(), "arch": "vit_small_patch16_224",
                    "img_size": args.img_size, "mask_ratio": args.mask_ratio,
                    "epoch": ep + 1, "epochs": args.epochs}
            torch.save(ckpt, f"{args.out}.ep{ep+1}")
            print(f"[SAVE] {args.out}.ep{ep+1}", flush=True)

    enc_sd = {k: v for k, v in model.state_dict().items()
              if not k.startswith(("decoder_", "mask_token"))}
    ckpt = {"model": enc_sd, "arch": "vit_small_patch16_224",
            "img_size": args.img_size, "mask_ratio": args.mask_ratio,
            "epoch": args.epochs}
    torch.save(ckpt, args.out)
    print(f"MAE_DONE {args.out}  encoder_keys={len(enc_sd)}", flush=True)


if __name__ == "__main__":
    main()
