from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_mode_groups(source: Path) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AFFINE_MODE_GROUPS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("AFFINE_MODE_GROUPS was not found in hierarchy.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the complete nine-mode affine traction interface space."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/adaptive_hierarchy/adaptive_report.json"),
    )
    parser.add_argument(
        "--hierarchy-source",
        type=Path,
        default=Path("ddpnm3d/hierarchy.py"),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    ninterfaces = int(report["counts"]["interfaces"])
    mode_groups = read_mode_groups(args.hierarchy_source)
    per_interface = tuple(
        mode for group in mode_groups.values() for mode in group
    )
    assert {name: len(group) for name, group in mode_groups.items()} == {
        "constant_vector": 3,
        "linear_normal": 2,
        "linear_tangential": 4,
    }
    expected_per_interface = len(per_interface)
    assert expected_per_interface == 9
    assert len(set(per_interface)) == 9
    keys = tuple(
        (interface_id, *mode)
        for interface_id in range(ninterfaces)
        for mode in per_interface
    )
    for interface_id in range(ninterfaces):
        local = [key[1:] for key in keys if key[0] == interface_id]
        assert len(local) == 9
        assert len(set(local)) == 9

    system = report["systems"]["HODDPNM"]
    error = report["strict_errors_to_identical_mesh_FEM"]["HODDPNM"]
    assert int(system["interface_unknowns"]) == 9 * ninterfaces
    assert float(system["relative_linear_residual"]) < 1.0e-10
    assert float(system["maximum_interface_moment_residual"]) < 1.0e-10
    assert float(system["relative_mass_imbalance"]) < 1.0e-9
    assert bool(error["same_parent_mesh_object"])
    print(
        "HODDPNM-P1(9) audit passed: "
        f"{ninterfaces} interfaces x 9 = {len(keys)} unknowns; "
        f"L2(u)={error['velocity_relative_l2']:.2%}, "
        f"broken-H1(u)={error['velocity_relative_broken_h1_seminorm']:.2%}, "
        f"flow={error['outlet_flux_relative_error']:.2%}."
    )


if __name__ == "__main__":
    main()
