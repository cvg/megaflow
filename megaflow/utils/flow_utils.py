import torch
import numpy as np
from scipy import interpolate


def merge_flows(flow1, valid1, flow2, valid2, method="nearest"):
    flow1 = np.transpose(flow1, axes=[2, 0, 1])

    _, ht, wd = flow1.shape

    x1, y1 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1_f = x1 + flow1[0]
    y1_f = y1 + flow1[1]

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x1_f = x1_f.reshape(-1)
    y1_f = y1_f.reshape(-1)
    valid1 = valid1.reshape(-1)

    mask1 = (
        (valid1 > 0.5) & (x1_f >= 0) & (x1_f <= wd - 1) & (y1_f >= 0) & (y1_f <= ht - 1)
    )
    x1 = x1[mask1]
    y1 = y1[mask1]
    x1_f = x1_f[mask1]
    y1_f = y1_f[mask1]
    valid1 = valid1[mask1]

    # STEP 1: interpolate valid values
    new_valid1 = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        valid2,
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    valid1 = new_valid1.round()

    mask1 = valid1 > 0.5
    x1 = x1[mask1]
    y1 = y1[mask1]
    x1_f = x1_f[mask1]
    y1_f = y1_f[mask1]
    valid1 = valid1[mask1]

    flow2_filled = fill_invalid(flow2, valid2)

    # STEP 2: interpolate flow values
    flow_x = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        flow2_filled[:, :, 0],
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    flow_y = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        flow2_filled[:, :, 1],
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )

    new_flow_x = np.zeros_like(flow1[0])
    new_flow_y = np.zeros_like(flow1[1])
    new_flow_x[(y1, x1)] = flow_x + x1_f - x1
    new_flow_y[(y1, x1)] = flow_y + y1_f - y1

    new_flow = np.stack([new_flow_x, new_flow_y], axis=0)

    new_valid = np.zeros_like(flow1[0])
    new_valid[(y1, x1)] = valid1

    new_flow = np.transpose(new_flow, axes=[1, 2, 0])
    return new_flow, new_valid


def fill_invalid(flow, valid):
    flow = np.transpose(flow, axes=[2, 0, 1])
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0.copy()
    y1 = y0.copy()

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    valid_flat = valid.reshape(-1)

    mask = valid_flat > 0.5
    x1 = x1[mask]
    y1 = y1[mask]
    dx = dx[mask]
    dy = dy[mask]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    flow = np.transpose(flow, axes=[1, 2, 0])
    return flow