import torch


def cap_distribution(p: torch.Tensor, p_max: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Caps each probability in p by p_max, redistributing excess
    proportionally among the remaining samples. Repeats until stable.

    Args:
        p (torch.Tensor): 1D tensor of probabilities (must sum to 1, ideally).
        p_max (float): Maximum allowed probability for any entry.
        eps (float): Small number to avoid numerical issues.

    Returns:
        torch.Tensor: New probability distribution with no element above p_max.
    """
    # Copy to avoid modifying input
    p_out = p.clone()

    while True:
        # Find indices above p_max
        over_max = p_out >= p_max
        if not over_max.any():
            break  # No probabilities exceed p_max
        if over_max.all():
            p_out = torch.ones_like(p_out) / p_out.numel()
            if p_out.sum() != 1:
                raise ValueError("P_max does not allow for any distribution")
            break

        # Calculate the total mass in these indices
        over_sum = p_out[over_max].sum()
        # Calculate how much we need to redistribute
        D = over_sum - over_max.sum() * p_max

        # If D is tiny or negative, no further redistribution is needed
        if D <= eps:
            break

        # Cap probabilities at p_max
        p_out[over_max] = p_max

        # Redistribute the excess among the rest
        under_mask = ~over_max
        under_sum = p_out[under_mask].sum()
        if under_sum > eps:
            # Distribute proportionally to the current masses
            p_out[under_mask] += (p_out[under_mask] / under_sum) * D
        else:
            # If everything is over the maximum, we can only cap
            # and break out.
            break

    return p_out


# Example usage:
if __name__ == "__main__":
    # Suppose p is a tensor of probabilities
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])  # sums to 1
    p_max = 0.2
    new_p = cap_distribution(p, p_max)
    print("Original distribution:", p)
    print("New distribution:", new_p)