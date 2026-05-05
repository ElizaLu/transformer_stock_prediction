import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import calc_ic, zscore, drop_extreme, DailyBatchSamplerRandom


class SequenceTrainer:
    def __init__(self, model, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU is not None else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.fitted = -1

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        return DataLoader(data, sampler=sampler, drop_last=drop_last) # smapler每次返回index

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            # 把所有参数的梯度强行限制在 [-3, 3] 之间（防止梯度爆炸）
            self.optimizer.step()

        return float(np.mean(losses))

    def predict(self, dl_test):
        if self.fitted < 0:
            raise ValueError("model is not fitted yet!")

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic_list = []
        ric_list = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred.ravel())
            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic_list.append(daily_ic)
            ric_list.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        metrics = {
            "IC": np.mean(ic_list),
            "ICIR": np.mean(ic_list) / np.std(ic_list),
            # ICIR（Information Coefficient Information Ratio），类似sharpe ratio
            "RIC": np.mean(ric_list),
            "RICIR": np.mean(ric_list) / np.std(ric_list),
        }
        return predictions, metrics

    def fit(self, dl_train, dl_valid=None, save_path="model", save_prefix="", seed=0):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = epoch

            if dl_valid is not None:
                _, metrics = self.predict(dl_valid)
                print(
                    f"Epoch {epoch}, train_loss {train_loss:.6f}, "
                    f"valid ic {metrics['IC']:.4f}, icir {metrics['ICIR']:.3f}, "
                    f"rankic {metrics['RIC']:.4f}, rankicir {metrics['RICIR']:.3f}."
                )
            else:
                print(f"Epoch {epoch}, train_loss {train_loss:.6f}")

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                torch.save(copy.deepcopy(self.model.state_dict()), save_path / f"{save_prefix}_{seed}.pkl")
                break

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 0