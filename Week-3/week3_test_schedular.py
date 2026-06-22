import torch
import pytest

from week3_schedular import NoiseScheduler


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_buffer_shapes(schedule):
    T = 1000
    ns = NoiseScheduler(timesteps=T, schedule=schedule)
    assert ns.betas.shape == (T,)
    assert ns.alphas.shape == (T,)
    assert ns.alphas_cumprod.shape == (T,)
    assert ns.sqrt_alphas_cumprod.shape == (T,)
    assert ns.sqrt_one_minus_alphas_cumprod.shape == (T,)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_betas_in_valid_range(schedule):
    ns = NoiseScheduler(timesteps=1000, schedule=schedule)
    assert (ns.betas > 0).all()
    assert (ns.betas < 1).all()


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alpha_equals_one_minus_beta(schedule):
    ns = NoiseScheduler(timesteps=1000, schedule=schedule)
    assert torch.allclose(ns.alphas, 1.0 - ns.betas)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alphas_cumprod_is_monotonically_decreasing(schedule):
    ns = NoiseScheduler(timesteps=1000, schedule=schedule)
    diffs = ns.alphas_cumprod[1:] - ns.alphas_cumprod[:-1]
    assert (diffs <= 1e-8).all(), "alpha_bar_t should be non-increasing in t"


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_add_noise_preserves_shape(schedule):
    ns = NoiseScheduler(timesteps=1000, schedule=schedule)
    x0 = torch.rand(8, 1, 28, 28) * 2 - 1
    t = torch.randint(0, 1000, (8,))
    x_t, noise = ns.add_noise(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_add_noise_at_t0_is_almost_clean():
    """At t=0, x_t should be nearly identical to x_0 — catches a reversed index bug."""
    ns = NoiseScheduler(timesteps=1000, schedule="linear")
    x0 = torch.rand(4, 1, 28, 28) * 2 - 1
    x_t, _ = ns.add_noise(x0, 0)
    assert torch.allclose(x_t, x0, atol=0.05)


def test_x_T_is_approximately_standard_normal():
    """Core correctness check: x_T should look like N(0, I) regardless of x_0."""
    ns = NoiseScheduler(timesteps=1000, schedule="linear")
    torch.manual_seed(0)
    x0 = torch.rand(4096, 1, 28, 28) * 2 - 1
    x_T, _ = ns.add_noise(x0, ns.timesteps - 1)

    mean = x_T.mean().item()
    std = x_T.std().item()
    assert abs(mean) < 0.05, f"x_T mean should be ~0, got {mean}"
    assert abs(std - 1.0) < 0.05, f"x_T std should be ~1, got {std}"


def test_cosine_and_linear_schedules_differ():
    linear = NoiseScheduler(timesteps=1000, schedule="linear")
    cosine = NoiseScheduler(timesteps=1000, schedule="cosine")
    assert not torch.allclose(linear.betas, cosine.betas)


def test_invalid_schedule_raises():
    with pytest.raises(ValueError):
        NoiseScheduler(timesteps=1000, schedule="not_a_real_schedule")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))