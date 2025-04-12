from enum import Enum
from typing import Dict


class HighlightRule(Enum):
    MAX = "max"
    MIN = "min"
    NONE = "none"


def get_highlight_rules() -> Dict[str, HighlightRule]:
    return {
        "log_name": HighlightRule.NONE,
        "ari": HighlightRule.MAX,
        "ami": HighlightRule.MAX,
        "completeness": HighlightRule.MAX,
        "homogeneity": HighlightRule.MAX,
        "nmi": HighlightRule.MAX,
        "v_measure": HighlightRule.MAX,
        "accuracy": HighlightRule.MAX,
        "time": HighlightRule.MIN,
        "inertia": HighlightRule.MIN,
    }
