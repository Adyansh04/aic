#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


RAIL_KEYS = [
    "nic_rail_0", "nic_rail_1", "nic_rail_2", "nic_rail_3", "nic_rail_4",
    "sc_rail_0", "sc_rail_1",
    "lc_mount_rail_0", "sfp_mount_rail_0", "sc_mount_rail_0",
    "lc_mount_rail_1", "sfp_mount_rail_1", "sc_mount_rail_1",
]


def load_base(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Base engine config not found: {path}")


def empty_task_board(board_pose: dict) -> dict:
    tb = {"pose": board_pose}
    for key in RAIL_KEYS:
        tb[key] = {"entity_present": False}
    return tb


def pose(x: float, y: float, z: float = 1.14, yaw: float = 3.1415) -> dict:
    return {"x": float(x), "y": float(y), "z": float(z), "roll": 0.0, "pitch": 0.0, "yaw": float(yaw)}


def rail(entity_name: str, translation: float, yaw: float = 0.0) -> dict:
    return {
        "entity_present": True,
        "entity_name": entity_name,
        "entity_pose": {
            "translation": float(translation),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(yaw),
        },
    }


def sfp_cable(cable_name: str = "cable_0") -> dict:
    return {
        cable_name: {
            "pose": {
                "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04245},
                "roll": 0.4432,
                "pitch": -0.4838,
                "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable",
        }
    }


def sc_cable(cable_name: str = "cable_0") -> dict:
    return {
        cable_name: {
            "pose": {
                "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04045},
                "roll": 0.4432,
                "pitch": -0.4838,
                "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable_reversed",
        }
    }


def make_sfp_trial(idx: int, board_x: float, board_y: float, yaw: float, nic_idx: int, nic_trans: float, nic_yaw: float, port_name: str) -> dict:
    tb = empty_task_board(pose(board_x, board_y, yaw=yaw))
    tb[f"nic_rail_{nic_idx}"] = rail(f"nic_card_{nic_idx}", nic_trans, nic_yaw)
    # Keep non-target clutter off for easier first dataset. Add clutter later after labels are verified.
    cable_name = "cable_0"
    return {
        "scene": {
            "task_board": tb,
            "cables": sfp_cable(cable_name),
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": port_name,
                "target_module_name": f"nic_card_mount_{nic_idx}",
                "time_limit": 300,
            }
        },
    }


def make_sc_trial(idx: int, board_x: float, board_y: float, yaw: float, sc_idx: int, sc_trans: float) -> dict:
    tb = empty_task_board(pose(board_x, board_y, yaw=yaw))
    tb[f"sc_rail_{sc_idx}"] = rail(f"sc_mount_{sc_idx}", sc_trans, 0.0)
    cable_name = "cable_0"
    return {
        "scene": {
            "task_board": tb,
            "cables": sc_cable(cable_name),
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{sc_idx}",
                "time_limit": 300,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AIC engine config for keypoint dataset collection.")
    parser.add_argument("--base", type=Path, default=Path("/ws_aic/install/share/aic_engine/config/sample_config.yaml"), help="Base engine config to copy scoring/task_board_limits/robot from")
    parser.add_argument("--out", type=Path, required=True, help="Output YAML path")
    parser.add_argument("--mode", choices=["debug10", "medium40"], default="debug10")
    args = parser.parse_args()

    base = load_base(args.base)
    out = {
        "scoring": deepcopy(base.get("scoring", {})),
        "task_board_limits": deepcopy(base.get("task_board_limits", {})),
        "trials": {},
        "robot": deepcopy(base.get("robot", {})),
    }

    # Conservative board poses near normal engine distribution. If a particular pose causes elbow/collision,
    # edit board_x/board_y/yaw here and regenerate.
    sfp_specs = [
        # board_x, board_y, yaw, nic_idx, nic_translation, nic_yaw, port_name
        (0.150, -0.200, 2.80, 0, -0.018, -0.05, "sfp_port_0"),
        (0.155, -0.190, 3.00, 1, -0.010,  0.00, "sfp_port_1"),
        (0.145, -0.205, 3.14, 2,  0.000,  0.05, "sfp_port_0"),
        (0.160, -0.185, 3.30, 3,  0.010, -0.05, "sfp_port_1"),
        (0.150, -0.195, 3.48, 4,  0.020,  0.05, "sfp_port_0"),
    ]
    sc_specs = [
        # board_x, board_y, yaw, sc_idx, sc_translation
        (0.170, 0.000, 2.80, 0, -0.050),
        (0.165, 0.015, 3.00, 1, -0.030),
        (0.175, -0.010, 3.14, 0,  0.000),
        (0.170, 0.020, 3.30, 1,  0.030),
        (0.160, -0.020, 3.48, 1,  0.050),
    ]

    specs = []
    if args.mode == "debug10":
        for spec in sfp_specs:
            specs.append(("sfp", spec))
        for spec in sc_specs:
            specs.append(("sc", spec))
    else:
        # Make 40 trials by repeating the 10 debug poses with small deterministic offsets.
        for rep in range(4):
            dx = (rep - 1.5) * 0.004
            dy = (rep % 2 - 0.5) * 0.006
            dyaw = (rep - 1.5) * 0.04
            for spec in sfp_specs:
                bx, by, yaw, ni, nt, nyaw, pn = spec
                specs.append(("sfp", (bx + dx, by + dy, yaw + dyaw, ni, max(-0.021, min(0.023, nt + dx)), nyaw, pn)))
            for spec in sc_specs:
                bx, by, yaw, si, st = spec
                specs.append(("sc", (bx + dx, by + dy, yaw + dyaw, si, max(-0.058, min(0.053, st + 2 * dx)))))

    for i, (kind, spec) in enumerate(specs, start=1):
        if kind == "sfp":
            out["trials"][f"trial_{i}"] = make_sfp_trial(i, *spec)
        else:
            out["trials"][f"trial_{i}"] = make_sc_trial(i, *spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"Wrote {args.out} with {len(out['trials'])} trials")


if __name__ == "__main__":
    main()
