"""End-to-end loader config benchmark: real GPU steps + given loader."""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Kevin\Desktop\glaucoma_first")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from train import FundusDS, load_rows, FundusNet, ThreadLoader, ModelEMA


def run(nthreads, bs=24, steps=60, use_ema=True):
    rows = load_rows("train")[:6000]
    ds = FundusDS("train", rows, 384, train=True)
    dl = ThreadLoader(ds, bs, nthreads=nthreads, shuffle=True, drop_last=True)
    m = FundusNet("efficientnet_b3", pretrained=False).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.BCEWithLogitsLoss()
    ema = ModelEMA(m, 0.999) if use_ema else None
    it = iter(dl)
    t0 = time.time()
    n = 0
    for s in range(steps):
        x, age, y = next(it)
        x, age, y = x.cuda(), age.cuda(), y.cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(m(x, age).float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ema:
            ema.update(m)
        n += len(y)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"threads={nthreads} bs={bs} ema={use_ema}: {n} imgs in {dt:.1f}s -> {n/dt:.1f} img/s "
          f"(epoch est {51396/(n/dt)/60:.1f} min)", flush=True)
    del m, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for nt in (4, 6, 8):
        run(nt)
    run(6, use_ema=False)