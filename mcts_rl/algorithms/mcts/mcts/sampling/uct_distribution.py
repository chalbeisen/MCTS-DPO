import torch


def puct_distribution(puct_values: torch.Tensor, inf_value_cap_coeff: float = 10.0) -> torch.Tensor:
    """
    PUCT distribution. PUCT values can be any real number (e.g. -inf).
    """
    # treat infinities
    sorted_puct = puct_values.sort().values
    infinite_puct = sorted_puct.abs() == float("inf")
    ### get max puct value (ignore inf values for that step)
    largest_finite_puct = sorted_puct[~infinite_puct].abs().max()
    ### normalize according to max puct value (the max not infinite value is set to 1 and the others are scaled accordingly)
    capped_puct = sorted_puct.clone() / largest_finite_puct
    ### update inf values to inf_value_cap_coeff
    capped_puct[infinite_puct] = inf_value_cap_coeff * torch.sign(sorted_puct[infinite_puct])

    # softmax
    puct_probabilities = torch.softmax(capped_puct, dim=0)
    ### normalize so probabilities sum up to 1 ?
    puct_probabilities = puct_probabilities / puct_probabilities.sum()

    return puct_probabilities


if __name__ == "__main__":
    from cap_distribution import cap_distribution

    puct_values = torch.tensor([-float("inf"), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")])
    dist = puct_distribution(puct_values, 2)
    print(dist)
    res = cap_distribution(dist, 0.5)
    print(res)
