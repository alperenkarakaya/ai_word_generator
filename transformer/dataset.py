"""
mmap tabanlı eğitim dataset'i.

train.bin / val.bin uint16 vocab id'lerinin ham dizisidir.
Her batch rastgele bir başlangıç indexi seçer ve block_size+1 token okur:
  x = ids[i : i+block_size]
  y = ids[i+1 : i+block_size+1]
"""
import os

import numpy as np
import torch


class BinDataset:
    """Sonsuz stream — DataLoader gerektirmez, doğrudan get_batch çağrılır."""

    def __init__(self, path: str, block_size: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
        self.path = path
        self.block_size = block_size
        # mmap ile aç — RAM'e tüm veriyi yüklemez.
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < block_size + 2:
            raise ValueError(f"Veri çok küçük ({len(self.data)} token).")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size: int, device: str | torch.device = "cpu"):
        n = len(self.data) - self.block_size - 1
        ix = np.random.randint(0, n, size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
