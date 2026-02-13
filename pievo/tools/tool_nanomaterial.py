import json
import requests
from typing import Dict, Any, List, Optional, Union, Annotated
import random

from . import register_tool


@register_tool(
    name="characterize_nanohelix_gfactor",
    description="Characterize the g-factor (dissymmetry factor) of a nanohelix structure based on its geometric parameters and incident light conditions. This tool predicts the chiral optical response using a machine-learned correlation derived from FDTD simulations.",
)
def characterize_nanohelix_gfactor(
    fiber_radius: Annotated[
        float,
        "Required field. This is the Thickness of the Wire. If you cut the wire, this is the radius of that circular cross-section. It determines how 'fat' the filament is. This parameter is not the same with the helix_radius. Both are required. "
    ],
    helix_radius: Annotated[
        float,
        "Required field. This is the Wideness of the Coil. It is the distance from the empty center of the spring to the center of the wire. It determines how large the overall cylinder is. This parameter is not the same with the fiber_radius. Both are required. "
    ],
    n_turns: Annotated[
        float,
        "Required field. The total number of complete helical turns (loops) in the structure."
    ],
    pitch: Annotated[
        float,
        "Required field. The axial periodic distance for one complete helical turn (in nm). This is a critical parameter for axial elongation and chiral strength."
    ],
    wavelength: Annotated[
        float,
        "Required field. The specific optical wavelength (in nm) of the incident light at which the g-factor is queried."
    ],
    x_y_z: Annotated[
        int,
        "Required field. Specifies the principal coordinate axis for the incident light source, where 0=x, 1=y, and 2=z axes."
    ] = 2,
    forward_backward: Annotated[
        int,
        "Required field. Indicates the polarity of light propagation along the chosen axis. 0 represents the Positive (Forward) direction, and 1 represents the Negative (Backward) direction."
    ] = 0
) -> Dict[str, Any]:
    """
    Predicts the g-factor (dissymmetry factor) of a nanohelix based on its geometric parameters.

    The g-factor quantifies the different optical responses of nanohelices to left and right
    circularly polarized light. Higher g-factor values indicate stronger chiral optical activity.

    Args:
        fiber_radius: Radius of the actual fiber/wire that forms the helix (in nm)
        helix_radius: Radius of the helix - distance from central axis to the center of the helical path (in nm)
        n_turns: Number of complete turns in the helix
        pitch: Axial distance between adjacent turns (in nm)
        wavelength: Wavelength of the helix.
        x_y_z: Incident light source (0=x, 1=y, 2=z)
        forward_backward: Indicates the polarity of light propagation along the chosen axis. 0 represents the Positive (Forward) direction, and 1 represents the Negative (Backward) direction.

    Returns:
        Dictionary containing the prediction result with the g-factor value
    """
    # Construct input data
    input_data = {
        "fiber_radius": fiber_radius,
        "helix_radius": helix_radius,
        "n_turns": n_turns,
        "pitch": pitch,
        "wavelength": wavelength,
        "x_y": x_y_z,
        "direction": forward_backward
    }

    # Call the API via Http requests. 
    try:
        
        ############################################################################
        #  NOTE: we only demonstrate the exact format of input/output in this tool, 
        #        as the surrogate model itself integrates some other packages.  
        ############################################################################

        return {
            "tool_name": "characterize_nanohelix_gfactor",
            "input": input_data,
            "output": random.uniform(0, 1),
            "success": True,
            "error": "[CRITICAL] Current tool call is for demo purpose (directly used `random.uniform(0, 1)` here), actual surrogate models or external wet lab devices should be integrated externally. For more help information, please see the official repo of PiEvo. ",
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "tool_name": "characterize_nanohelix_gfactor",
            "input": input_data,
            "output": None,
            "success": False,
            "error": f"API request failed: {str(e)}",
        }
    except Exception as e:
        return {
            "tool_name": "characterize_nanohelix_gfactor",
            "input": input_data,
            "output": None,
            "success": False,
            "error": f"Error processing request: {str(e)}",
        }
