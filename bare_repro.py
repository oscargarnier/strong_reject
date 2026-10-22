import torch, torch.nn.functional as F
print(torch.__version__, torch.version.hip)
for dt in (torch.float32, torch.bfloat16):
    x = torch.randn(1, 8192, 64, device="cuda", dtype=dt)
    w = torch.randn(8192, 1, 4, device="cuda", dtype=dt)
    y = F.conv1d(x, w, None, padding=3, groups=8192)
    torch.cuda.synchronize()
    print(dt, "ok", y.shape)
