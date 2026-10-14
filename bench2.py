import os
import sys
import time

sys.path.insert(0, r"C:\Users\Kevin\Desktop\glaucoma_first")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from train import FundusNet


def bench(arch, size, bs, dtype, channels_last, iters=12, accum=1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = FundusNet(arch, pretrained=False).cuda()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(bs, 3, size, size, device="cuda")
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    age = torch.randn(bs, 1, device="cuda")
    y = torch.rand(bs, device="cuda")
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = crit(m(x, age).float(), y) / accum
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = crit(m(x, age).float(), y) / accum
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    st = (time.time() - t0) / iters
    vram = torch.cuda.max_memory_allocated() / 2**30
    tag = f"{arch}@{size} bs{bs} {'fp16' if dtype == torch.float16 else 'bf16'}{' cl' if channels_last else ''}"
    print(f"{tag:42s} {st*1000:6.0f} ms/step  {bs/st:5.0f} img/s  vram {vram:.2f}GB", flush=True)
    del m, opt, x, age, y
    torch.cuda.empty_cache()


def main():
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    bench("efficientnet_b3", 384, 32, torch.float16, False)
    bench("efficientnet_b3", 384, 32, torch.float16, True)
    bench("efficientnet_b3", 384, 32, torch.bfloat16, True)
    bench("efficientnet_b3", 384, 24, torch.float16, True)
    bench("efficientnet_b3", 384, 48, torch.float16, True)
    bench("efficientnet_b3", 320, 32, torch.float16, True)
    bench("efficientnet_b4", 384, 24, torch.float16, True)
    bench("efficientnet_b4", 448, 16, torch.float16, True)


if __name__ == "__main__":
    main()