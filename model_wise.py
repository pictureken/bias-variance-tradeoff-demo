import matplotlib.pyplot as plt
import numpy as np

import utils


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
        result_dict = utils.calculate.task(
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
    plt.savefig(f"./outputs/linear-regresiion/{is_fix}/bias_variance_tradeoff.png")


if __name__ == "__main__":
    TOY_SAMPLE_SIZE = 100
    STD = 1.0
    TEST_SAMPLE_SIZE = 50
    TRIAL = 100
    degree_list = list(range(1, 11))
    main(
        degree_list=degree_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=False,
    )
    main(
        degree_list=degree_list,
        toy_sample_size=TOY_SAMPLE_SIZE,
        std=STD,
        test_sample_size=TEST_SAMPLE_SIZE,
        trial=TRIAL,
        is_fix=True,
    )
