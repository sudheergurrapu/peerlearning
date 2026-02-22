from dataclasses import dataclass


@dataclass(frozen=True)
class PricingConfig:
    min_threshold: float = 0.70
    max_threshold: float = 1.50
    max_step_change: float = 0.10


CONFIG = PricingConfig()
