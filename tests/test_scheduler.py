import math
import pytest
from myproject.scheduler.scheduler import CosineAnnealingLR


class FakeOptimizer:
    def __init__(self, lr):
        self.lr = lr


def test_cosine_schedule_values():
    opt = FakeOptimizer(0.1)
    sched = CosineAnnealingLR(opt, num_epochs = 4, min_lr = 0.0)
    lrs = []
    for _ in range(4):
        sched.step(1.0)
        lrs.append(opt.lr)
    expected = [0.1 * (1 + math.cos(math.pi * t / 4)) / 2 for t in range(1, 5)]
    assert lrs == pytest.approx(expected)
    assert lrs[1] == pytest.approx(0.05)
    assert lrs[-1] == pytest.approx(0.0)


def test_cosine_stays_at_min_after_end():
    opt = FakeOptimizer(0.1)
    sched = CosineAnnealingLR(opt, num_epochs = 2, min_lr = 1e-3)
    for _ in range(5):
        sched.step()
    assert opt.lr == pytest.approx(1e-3)
