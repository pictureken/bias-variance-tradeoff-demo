from typing import List

import matplotlib.pyplot as plt
import numpy as np

import utils


def main(
    hidden_size_list: List[int],
    toy_sample_size: int,
    std: float,
    test_sample_size: int,
    trial: int,
    is_fix: bool = False,
):
    bias_variance_list = {"mse": [], "bias2": [], "variance": []}
    for d in hidden_size_list:
        np.random.seed(2023)
        result_dict = utils.calculate_nn.task(
            toy_sample_size=toy_sample_size,
            std=std,
            test_sample_size=test_sample_size,
            trial=trial,
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
    # plt.ylim([0, 0.3])
    plt.xscale("log")
    plt.xlabel("Number of hidden units")
    plt.ylabel("MSE/Bias2/Variance")
    plt.savefig(f"./outputs/nn/{is_fix}/bias_variance_tradeoff.png")


if __name__ == "__main__":
    TOY_SAMPLE_SIZE = 100
    STD = 1.0
    TEST_SAMPLE_SIZE = 50
    TRIAL = 100
    hidden_size_list = [5, 10, 15, 17, 20, 22, 25, 35, 75, 100, 1000, 10000]
    hidden_size_list = [35]
    main(
        hidden_size_list=hidden_size_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=False,
    )
    main(
        hidden_size_list=hidden_size_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=True,
    )
