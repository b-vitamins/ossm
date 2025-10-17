from __future__ import annotations
import itertools
import torch
from torch.utils.data import DataLoader, TensorDataset
from ossm.data.datasets.loader_utils import infinite


def test_infinite_wrapper():
    ds = TensorDataset(torch.arange(10))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    it = infinite(loader)
    got = list(itertools.islice(it, 3))
    sizes = [b[0].shape[0] for b in got]
    assert sizes == [4, 4, 2]
