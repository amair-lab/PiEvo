import hashlib
import json
import math
from typing import Any, Optional, Dict, Sequence


def sanitize(
        value: Any,
        low: float = 0.0,
        high: float = 1.0
) -> Optional[float]:
    """
    Sanitize and validate prediction values to be within bounds.
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(v):
        return None
    return float(max(low, min(high, v)))



def get_submission_hash(submission: Dict[str, Any]) -> str:
    content_to_hash = f"{submission['source_agent']}:{json.dumps(submission['json_data'], sort_keys=True)}"
    return hashlib.md5(content_to_hash.encode()).hexdigest()



def entropy_of_probs(probs: Sequence[float]) -> float:
    """
    Compute Shannon entropy of probability distribution.
    """
    eps = 1e-12
    return -sum(p * math.log(max(eps, p)) for p in probs if p > eps)