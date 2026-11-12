def get_brownian_bridge(
    args,
    x_pairs: torch.Tensor,
    t_pairs: torch.Tensor,
    direction: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates a Brownian bridge between t1 and t2 with z0 at t1 and z1 at t2.

    Args:
        args: object with attributes args.sigma, args.eps, and args.accelerator.device
        x_pairs: Tensor of shape [batch, 2, dim] containing (z0, z1)
        t_pairs: Tensor of shape [2] containing (t1, t2)
        direction: "forward" or "backward"

    Returns:
        z_t: sample from the Brownian bridge at time t
        t: sampled time tensor [batch, 1]
        target: estimated score (∇ log p(x_t))
    """

    device = args.accelerator.device
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]  # values at t1 and t2
    t1, t2 = t_pairs[:, 0], t_pairs[:, 1]  # time bounds reshaped to [batch, 1]

    t1 = t1.unsqueeze(1)  # [num_samples] -> [num_samples, 1]
    t2 = t2.unsqueeze(1)

    # Sample t uniformly in (t1 + eps, t2 - eps) to avoid sqrt(0) issues

    t = torch.rand((z0.shape[0], 1), device=device)

    t = t1 + (t2 - t1) * ((1 - 2 * args.eps) * t + args.eps)

    # Normalize time to [0, 1] as s = (t - t1) / (t2 - t1)
    s = (t - t1) / (t2 - t1)

    t_pairs_bridges = args.t_pairs_bridges
    sigma = 1

    # Compute the mean of the Brownian bridge at time t
    z_t = (1 - s) * z0 + s * z1

    # Sample standard Gaussian noise
    z = torch.randn_like(z_t, device=device)
    sigma = args.coeff_sigma * sigma
    # Add the stochastic part of the Brownian bridge
    z_t = z_t + sigma * torch.sqrt(s * (1 - s)) * z

    # Estimate the score depending on direction
    match direction:
        case "forward":
            target = (z1 - z0) - sigma * torch.sqrt(s / (1 - s)) * z
        case "backward":
            target = -(z1 - z0) - sigma * torch.sqrt((1 - s) / s) * z
        case _:
            raise ValueError("Direction must be 'forward' or 'backward'.")

    return z_t, t, target, sigma