"""Point-collection GOSPA (p=2, alpha=2), not a flow/trajectory estimator.

Independent NumPy/SciPy implementation of Rahmathullah et al., FUSION 2017,
Proposition 1: https://arxiv.org/abs/1601.05585 . No author code is copied.
The component readout is a proposed measurement contract, not an official
occupancy benchmark. It uses no boxes, GT positions, future flow or tracking.
"""

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _locations(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite [N,2] xy locations in metres")
    return arr


def gospa2(truth_xy, prediction_xy, cutoff_m=4.0, max_pairwise_entries=4_000_000):
    """GOSPA on xy component locations, with explicit squared-cost breakdown.

    A miss or false component costs c^2/2; a matched pair costs distance^2.
    Matching at distance >= c is represented as one miss plus one false.
    Coincident component locations retain multiplicity (they are NOT deduped).
    Empty arrays must retain shape [0,2]. Units: distance m, costs m^2.

    The rectangular capped assignment is exactly the alpha=2 objective, not
    nearest-neighbour or greedy matching. If a budget would be exceeded, fail
    without dropping/filtering components. Equal-cost assignment decompositions
    can be non-unique; the scalar is invariant, but group attribution need not be.
    """
    truth = _locations(truth_xy, "truth_xy")
    pred = _locations(prediction_xy, "prediction_xy")
    c = float(cutoff_m)
    if not np.isfinite(c) or c <= 0. or not np.isfinite(c*c):
        raise ValueError("cutoff_m must have a finite positive square")
    if (isinstance(max_pairwise_entries, (bool, np.bool_))
            or not isinstance(max_pairwise_entries, (int, np.integer))
            or max_pairwise_entries < 1):
        raise ValueError("max_pairwise_entries must be a positive integer")
    n, m = len(truth), len(pred)
    if n*m > max_pairwise_entries:
        raise ValueError("Pairwise assignment budget exceeded; no components were dropped")
    matches = []
    matched_truth, matched_pred = set(), set()
    if n and m:
        distance_squared = cdist(truth, pred, metric="sqeuclidean")
        if not np.isfinite(distance_squared).all():
            raise ValueError("Pairwise distance overflow")
        costs = np.minimum(distance_squared, c*c)
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if distance_squared[i, j] < c*c:
                i, j = int(i), int(j)
                matches.append(dict(truth_index=i, prediction_index=j,
                                    squared_distance_m2=float(distance_squared[i, j])))
                matched_truth.add(i); matched_pred.add(j)
    missed = [i for i in range(n) if i not in matched_truth]
    false = [j for j in range(m) if j not in matched_pred]
    localization = sum(x["squared_distance_m2"] for x in matches)
    missed_cost, false_cost = len(missed)*c*c/2., len(false)*c*c/2.
    total = localization + missed_cost + false_cost
    return dict(schema="component-gospa2-core-v1", p=2, alpha=2,
                cutoff_m=c, distance_m=float(np.sqrt(total)), squared_cost_m2=float(total),
                localization_m2=float(localization), missed_m2=float(missed_cost),
                false_m2=float(false_cost), truth_count=n, prediction_count=m,
                matches=matches, missed_truth_indices=missed, false_prediction_indices=false,
                no_temporal_identity_or_flow_evaluated=True)


def truth_group_breakdown(result, truth_groups):
    """Partition a FULL matching; never rematch moving GT against all predictions.

    Static matches stay static. False predictions remain one global unassigned
    term, since their true motion group is unknown. These grouped terms are NOT
    independent GOSPA scores. Labels may include moving/stationary/mixed/unknown.
    """
    if (len(truth_groups) != result["truth_count"]
            or any(not isinstance(x, str) or not x for x in truth_groups)):
        raise ValueError("One nonempty string group per truth location is required")
    groups = {name: dict(truth_count=0, matched_count=0, missed_count=0,
                         localization_m2=0., missed_m2=0.) for name in sorted(set(truth_groups))}
    for name in truth_groups:
        groups[name]["truth_count"] += 1
    for pair in result["matches"]:
        group = groups[truth_groups[pair["truth_index"]]]
        group["matched_count"] += 1
        group["localization_m2"] += pair["squared_distance_m2"]
    for i in result["missed_truth_indices"]:
        group = groups[truth_groups[i]]
        group["missed_count"] += 1
        group["missed_m2"] += result["cutoff_m"]**2/2.
    return dict(groups=groups, global_false_count=len(result["false_prediction_indices"]),
                global_false_m2=result["false_m2"], independent_group_gospa=False)


def bev_component_centers(foreground_xyz, extent_xyz):
    """Fixed label-free readout: any occupied z -> 4-connected xy components.

    `foreground_xyz` is a boolean [X,Y,Z] array already produced by the SAME
    full-resolution occupancy/valid-mask contract for every prediction and GT.
    extent=[xmin,ymin,zmin,xmax,ymax,zmax] gives outer voxel boundaries.
    Every nonempty component is kept. No size pruning, closing, erosion, box
    cropping, GT-assisted splitting, height weighting, or temporal association.
    Returns uniform-BEV-cell centroid, BEV area count and the component map.
    Array index 0 maps to physical x; array index 1 maps to physical y.
    """
    fg = np.asarray(foreground_xyz)
    bounds = np.asarray(extent_xyz, dtype=np.float64)
    if fg.dtype != np.bool_ or fg.ndim != 3 or any(n < 1 for n in fg.shape):
        raise ValueError("foreground_xyz must be nonempty boolean [X,Y,Z]")
    if (bounds.shape != (6,) or not np.isfinite(bounds).all()
            or np.any(bounds[3:] <= bounds[:3])):
        raise ValueError("Invalid extent_xyz")
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    component_map, count = label(fg.any(axis=2), structure=structure)
    if count == 0:
        return dict(centers_xy=np.empty((0, 2), dtype=np.float64),
                    area_cells=np.empty(0, dtype=np.int64), component_map=component_map)
    x, y = np.nonzero(component_map)
    ids = component_map[x, y]
    areas = np.bincount(ids, minlength=count+1)[1:]
    mean_x = np.bincount(ids, weights=x.astype(np.float64)+.5, minlength=count+1)[1:]/areas
    mean_y = np.bincount(ids, weights=y.astype(np.float64)+.5, minlength=count+1)[1:]/areas
    spacing_xy = (bounds[3:5] - bounds[:2])/np.array(fg.shape[:2])
    centers = np.column_stack((mean_x, mean_y))*spacing_xy + bounds[:2]
    return dict(centers_xy=centers, area_cells=areas, component_map=component_map)
