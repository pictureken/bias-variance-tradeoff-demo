import contextlib
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Simulator:
    def __init__(self, is_fix: bool):
        # sin(2πx)
        self.func = lambda x: np.sin(2 * np.pi * x)
        # Trueの場合fix design. Falseの場合random design.
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
    def bias_variance_tradeoff(self, y_list: list[np.ndarray]):
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


def task(
    is_fix: bool,
    toy_sample_size: int = 100,
    std: float = 1.0,
    test_sample_size: int = 50,
    trial: int = 100,
    degree: int = 3,
):
    y_pred_list = []
    # Simulatorクラス呼び出し
    simulator = Simulator(is_fix=is_fix)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for i in range(trial):
        # モデルを定義
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression()),
            ]
        )
        # 学習データを生成
        simulator.create_toy_data(toy_sample_size=toy_sample_size, std=std)
        # 評価用データを生成
        simulator.create_test_data(test_sample_size=test_sample_size)
        # モデルの学習
        simulator.train(model)
        # 評価
        simulator.test()
        y_pred_list.append(simulator.y_pred)
        # 20個の予測結果をプロット
        if i < 20:
            plt.plot(
                simulator.x_test,
                simulator.y_pred.reshape(-1, 1),
                c="orange",
            )
    # 全ての予測結果の平均をプロット
    plt.plot(
        simulator.x_test, np.asarray(y_pred_list).mean(axis=0), c="green", label="mean"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1.5, 1.5)

    # 20個の予測結果の平均と真の関数をプロット
    plt.subplot(1, 2, 2)
    plt.plot(simulator.x_test, simulator.y_true, c="red", label="gt")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(
        simulator.x_test, np.asarray(y_pred_list).mean(axis=0), c="green", label="mean"
    )
    plt.ylim(-1.5, 1.5)
    plt.legend()
    os.makedirs(f"./outputs/linear-regresiion/{is_fix}", exist_ok=True)
    plt.savefig(f"./outputs/linear-regresiion/{is_fix}/d{degree}.png")
    plt.close()

    # 二乗誤差とそれに対応する偏りと分散の算出
    mse, bias2, variance = simulator.bias_variance_tradeoff(y_pred_list)

    return {"mse": mse, "bias2": bias2, "variance": variance}


if __name__ == "__main__":
    task(is_fix=False, degree=3)
    task(is_fix=True, degree=3)
