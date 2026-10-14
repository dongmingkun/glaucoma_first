import os
import sys
import time

sys.path.insert(0, r"C:\Users\Kevin\Desktop\glaucoma_first")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch.utils.data import DataLoader

from train import FundusDS, load_rows, FundusNet


def main():
    rows = load_rows("train")[:3200]
    for nw in (4, 6):
        ds = FundusDS("train", rows, 384, train=True)
        dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=nw, pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4)
        t0 = time.time()
        n = 0
        first = None
        for x, a, y in dl:
            if first is None:
                first = time.time() - t0
            n += len(y)
            if n >= 1600:
                break
        dt = time.time() - t0
        print(f"DL workers={nw}: {n} imgs in {dt:.1f}s = {n/dt:.0f} img/s (first batch {first:.1f}s)", flush=True)
        del dl, ds

    m = FundusNet("efficientnet_b3", pretrained=False).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(32, 3, 384, 384, device="cuda")
    age = torch.randn(32, 1, device="cuda")
    y = torch.rand(32, device="cuda")
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(m(x, age).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(m(x, age).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    st = (time.time() - t0) / 20
    print(f"B3@384 bs32: {st*1000:.0f} ms/step -> {32/st:.0f} img/s train, vram {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)

    m.eval()
    with torch.no_grad():
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                m(x, age)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                m(x, age)
        torch.cuda.synchronize()
        st = (time.time() - t0) / 20
    print(f"B3@384 bs32 inference: {st*1000:.0f} ms/step -> {st/32*1000:.1f} ms/img", flush=True)


if __name__ == "__main__":
    main()