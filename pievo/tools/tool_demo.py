import warnings
from typing import Annotated, Dict, Any


from . import register_tool

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------
@register_tool(
    name="evaluate_f_function",
    description=(""),
)
def evaluate_f_function(
    x: Annotated[
        float,
        (
            "Input x value for the F-function evaluation. This is a placeholder tool and does not perform actual predictions. "
        ),
    ],
    y: Annotated[
        float,
        (
            "Input y value for the F-function evaluation. This is a placeholder tool and does not perform actual predictions. "
        ),
    ] = "unknown",
) -> Dict[str, Any]:

    hidden_func = lambda x, y: x**2 + y**2

    return {
        "tool_name": "evaluate_f_function",
        "success": True,
        "x": x,
        "y": y,
        "outcome": hidden_func(x, y),
        "error": None,
    }
