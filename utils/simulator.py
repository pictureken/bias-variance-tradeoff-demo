import contextlib
from typing import List

import numpy as np
import torch


class Simulator:
    def __init__(self, is_fix: bool):
        # sin(2πx)
        self.func = lambda x: np.sin(2 * np.pi * x)
        # Trueの場合fixed design. Falseの場合random design.
        self.is_fix = is_fix

    # 学習データの生成
    def create_toy_data(self, toy_sample_size: int, std: float):
        if self.is_fix:
            # 学習データの生成時のみseedを固定することでxを固定（fix design）
            with self._temporary_seed(2023):
                self.x = np.random.uniform(low=0, high=1, size=toy_sample_size)

        else:
            # 一様分布からxを生成（random design）
            self.x = np.random.uniform(low=0, high=1, size=toy_sample_size)

        np.random.shuffle(self.x)
        # 正規分布からノイズを生成
        self.epsilon = np.random.normal(scale=std, size=self.x.shape)

        # y=f(x)+ε
        self.y = self.func(self.x) + self.epsilon

    # 評価データの生成
    def create_test_data(self, test_sample_size: int):
        self.x_test = np.linspace(start=0, stop=1, num=test_sample_size)
        self.y_true = self.func(self.x_test)

    # モデルの学習
    def train(self, model):
        self.model = model
        reshaped_x, reshaped_y = self.x.reshape(-1, 1), self.y.reshape(-1, 1)
        self.model.fit(reshaped_x, reshaped_y)

    # モデルによる評価
    def test(self):
        reshaped_x_test = self.x_test.reshape(-1, 1)
        self.y_pred = self.model.predict(reshaped_x_test)

    # 二乗誤差と偏りと分散を算出（bias-variance tradeoff）
    def bias_variance_tradeoff(self, y_list: List[np.ndarray]):
        y_list = np.asarray(y_list)
        reshaped_y_true = self.y_true.reshape(-1, 1)
        mse = ((reshaped_y_true - y_list) ** 2).mean()
        bias2 = ((reshaped_y_true - y_list.mean(axis=0)) ** 2).mean()
        variance = ((y_list - y_list.mean(axis=0)) ** 2).mean()
        return mse, bias2, variance

    # コンテキストマネージャを使った特定の範囲のseedの固定
    @contextlib.contextmanager
    def _temporary_seed(self, seed: int):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


class NeuralNetSimulator(Simulator):
    def train(self, model, optimizer, criterion, epoch):
        losses = []
        self.model = model
        tensor_x = torch.from_numpy(self.x.astype(np.float32)).float()
        tensor_x = torch.stack([torch.ones(tensor_x.shape), tensor_x], 1)
        tensor_y = torch.from_numpy(self.y.astype(np.float32)).float()
        for _ in range(epoch):
            optimizer.zero_grad()
            y_pred = self.model(tensor_x)
            loss = criterion(y_pred.reshape(tensor_y.shape), tensor_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test(self):
        tensor_x = torch.from_numpy(self.x_test.astype(np.float32)).float()
        tensor_x = torch.stack([torch.ones(tensor_x.shape), tensor_x], 1)
        with torch.no_grad():
            self.y_pred = self.model(tensor_x).cpu().data.numpy().T[0].reshape(-1, 1)
