import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils


def task(
    is_fix: bool,
    toy_sample_size: int,
    std: float,
    test_sample_size: int,
    trial: int,
    epoch: int,
    hidden_size: int,
):
    y_pred_list = []
    # Simulatorクラス呼び出し
    simulator = utils.simulator.NeuralNetSimulator(is_fix=is_fix)
    criterion = nn.MSELoss()

    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for i in tqdm(range(trial)):
        model = utils.model.NeuralNet(
            in_features=2, hidden_size=hidden_size, out_features=1
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # 学習データを生成
        simulator.create_toy_data(toy_sample_size=toy_sample_size, std=std)
        # 評価用データを生成
        simulator.create_test_data(test_sample_size=test_sample_size)
        # モデルの学習
        simulator.train(model, optimizer, criterion, epoch)
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
    os.makedirs(f"./outputs/nn/{is_fix}", exist_ok=True)
    plt.savefig(f"./outputs/nn/{is_fix}/d{hidden_size}.png")
    fig.tight_layout()
    plt.close()

    # 二乗誤差とそれに対応する偏りと分散の算出
    mse, bias2, variance = simulator.bias_variance_tradeoff(y_pred_list)

    return {"mse": mse, "bias2": bias2, "variance": variance}


def main(
    hidden_size_list: List[int],
    toy_sample_size: int,
    std: float,
    test_sample_size: int,
    trial: int,
    epoch: int,
    is_fix: bool = False,
):
    bias_variance_list = {"mse": [], "bias2": [], "variance": []}
    for d in hidden_size_list:
        print(f"hidden_size:{d}")
        np.random.seed(2023)
        result_dict = task(
            toy_sample_size=toy_sample_size,
            std=std,
            test_sample_size=test_sample_size,
            trial=trial,
            epoch=epoch,
            hidden_size=d,
            is_fix=is_fix,
        )
        [
            bias_variance_list[key].append(result_dict[key])
            for key in bias_variance_list.keys()
        ]

    plt.figure()
    [
        plt.plot(hidden_size_list, bias_variance_list[key], label=key)
        for key in bias_variance_list.keys()
    ]
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Number of hidden units")
    plt.ylabel("MSE/Bias2/Variance")
    plt.savefig(f"./outputs/nn/{is_fix}/bias_variance_tradeoff.png")


if __name__ == "__main__":
    # 学習データの枚数
    TOY_SAMPLE_SIZE = 100
    # εが従う正規分布の分散
    STD = 0.5
    # テストデータの枚数
    TEST_SAMPLE_SIZE = 50
    # シミュレートの回数
    TRIAL = 100
    # epoch数
    EPOCH = 2000
    # 隠れ層のユニット数
    hidden_size_list = [5, 10, 15, 17, 20, 22, 25, 35, 75, 100, 1000]
    main(
        hidden_size_list=hidden_size_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        epoch=EPOCH,
        is_fix=False,
    )
    main(
        hidden_size_list=hidden_size_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        epoch=EPOCH,
        is_fix=True,
    )
