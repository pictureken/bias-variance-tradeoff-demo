import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import utils


def task(
    is_fix: bool,
    toy_sample_size: int,
    std: float,
    test_sample_size: int,
    trial: int,
    degree: int,
):
    y_pred_list = []
    # Simulatorクラス呼び出し
    simulator = utils.simulator.Simulator(is_fix=is_fix)
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
    os.makedirs(f"./outputs/linear-reg/{is_fix}", exist_ok=True)
    plt.savefig(f"./outputs/linear-reg/{is_fix}/d{degree}.png")
    plt.close()

    # 二乗誤差とそれに対応する偏りと分散の算出
    mse, bias2, variance = simulator.bias_variance_tradeoff(y_pred_list)

    return {"mse": mse, "bias2": bias2, "variance": variance}


def main(
    degree_list: list[int],
    toy_sample_size: int,
    std: float,
    test_sample_size: int,
    trial: int,
    is_fix: bool = False,
):
    bias_variance_list = {"mse": [], "bias2": [], "variance": []}
    for d in degree_list:
        np.random.seed(2023)
        result_dict = task(
            toy_sample_size=toy_sample_size,
            std=std,
            test_sample_size=test_sample_size,
            trial=trial,
            degree=d,
            is_fix=is_fix,
        )
        [
            bias_variance_list[key].append(result_dict[key])
            for key in bias_variance_list.keys()
        ]

    plt.figure()
    [
        plt.plot(degree_list, bias_variance_list[key], label=key)
        for key in bias_variance_list.keys()
    ]
    plt.legend()
    plt.ylim([0, 0.3])
    plt.xlabel("Degree")
    plt.ylabel("MSE/Bias2/Variance")
    plt.savefig(f"./outputs/linear-reg/{is_fix}/bias_variance_tradeoff.png")


if __name__ == "__main__":
    # 学習データの枚数
    TOY_SAMPLE_SIZE = 100
    # εが従う正規分布の分散
    STD = 1.0
    # テストデータの枚数
    TEST_SAMPLE_SIZE = 50
    # シミュレートの回数
    TRIAL = 100
    # 多項式の次数
    DEGREE_LIST = list(range(1, 11))
    main(
        degree_list=DEGREE_LIST,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=False,
    )
    main(
        degree_list=DEGREE_LIST,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=True,
    )
