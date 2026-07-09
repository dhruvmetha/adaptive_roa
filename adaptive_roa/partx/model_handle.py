from __future__ import annotations


class GPModelHandle:
    """Engine-compatible wrapper around a fitted GPClassifier."""

    def __init__(self, gp, system):
        self.gp = gp
        self.system = system

    def eval(self):
        self.gp.eval()
        return self

    def to(self, device):
        self.gp.to(device)
        return self
