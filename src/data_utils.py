import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


def calc_ic(pred, label):
    """计算IC和Rank IC
    """
    df = pd.DataFrame({"pred": np.asarray(pred).ravel(), "label": np.asarray(label).ravel()})
    return df["pred"].corr(df["label"]), df["pred"].corr(df["label"], method="spearman")


def zscore(x):
    std = x.std()
    return x - x.mean() if std <= 1e-8 else (x - x.mean()) / std


def drop_extreme(x, ratio=0.025):
    n = x.shape[0]
    n_drop = int(ratio * n)
    if n_drop <= 0 or n <= 2 * n_drop:
        mask = torch.ones_like(x, dtype=torch.bool)
        return mask, x

    _, indices = x.sort()
    kept = indices[n_drop:-n_drop]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[kept] = True
    return mask, x[mask]


def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            order = np.arange(len(self.daily_count))
            np.random.shuffle(order)
            for i in order:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.daily_count)