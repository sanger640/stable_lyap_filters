"""Physical failure definitions for the three-block Jenga task."""
import numpy as np

TOPPLE_DEG = 45.0
TABLE_TOP_Z = 0.425
BLOCK_NAMES = ("middle", "left", "right")


def failure_masks(peak_tilt, min_position_z=None):
    """Return separate and combined per-probe failure masks.

    Controlled translation or lifting of the middle block is not failure.  It fails
    when it topples past 45 degrees or its centre falls below the tabletop plane.
    Either side block toppling past 45 degrees is a neighboring-block failure.
    """
    tilt = np.asarray(peak_tilt, np.float64)
    if tilt.shape[-1] != 3:
        raise ValueError("peak_tilt must end with [middle, left, right]")
    middle_topple = tilt[..., 0] >= TOPPLE_DEG
    neighbor_topple = np.max(tilt[..., 1:], axis=-1) >= TOPPLE_DEG
    if min_position_z is None:
        middle_fall = np.zeros_like(middle_topple)
    else:
        z = np.asarray(min_position_z, np.float64)
        middle_z = z[..., 0] if z.shape == tilt.shape else z
        middle_fall = middle_z < TABLE_TOP_Z
    middle_failure = middle_topple | middle_fall
    return {"middle_topple": middle_topple, "middle_fall": middle_fall,
            "middle_failure": middle_failure, "neighbor_failure": neighbor_topple,
            "any_failure": middle_failure | neighbor_topple}


def boundary_mask(outcomes, minimum_each=2):
    values = np.asarray(outcomes, bool)
    failures = values.sum(axis=-1)
    return (failures >= int(minimum_each)) & ((values.shape[-1] - failures) >= int(minimum_each))
