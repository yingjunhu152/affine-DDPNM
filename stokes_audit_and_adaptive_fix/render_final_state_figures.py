from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)

METHOD_COLOR = {
    "PNM": "#5c7cfa",
    "DDPNM": "#37b24d",
    "DDPNMT": "#f59f00",
    "HODDPNM": "#e03131",
}
METHOD_LEVEL = {name: i for i, name in enumerate(METHOD_COLOR)}


def render_final_state_figures(
    mesh,
    levels: np.ndarray,
    fields: dict[str, np.ndarray],
    dof_data: dict[str, np.ndarray],
    out_dir: Path,
    grid_size: int = 104,
) -> dict[str, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_shape = np.asarray(mesh.domain_shape, dtype=float) * mesh.voxel_size
    pore = pore_mask(mesh)
    geometry = build_reference_style_geometry(pore, mesh.voxel_size, domain_shape, grid_size)
    outputs: dict[str, str | None] = {}

    region_path = out_dir / "final_region_methods.png"
    render_final_region_methods(mesh, levels, dof_data, region_path)
    outputs["final_region_methods"] = str(region_path)

    scalar_specs: list[tuple[str, np.ndarray, str, str, str]] = []
    if "hodd_velocity" in fields:
        scalar_specs.append(
            (
                "final_velocity_magnitude",
                np.linalg.norm(np.asarray(fields["hodd_velocity"], dtype=float), axis=1),
                "velocity magnitude",
                "turbo",
                "linear",
            )
        )
    if "hodd_pressure" in fields:
        scalar_specs.append(
            (
                "final_pressure",
                np.asarray(fields["hodd_pressure"], dtype=float),
                "pressure",
                "turbo",
                "linear",
            )
        )
    if "velocity_error" in fields:
        scalar_specs.append(
            (
                "final_velocity_log_error",
                np.linalg.norm(np.asarray(fields["velocity_error"], dtype=float), axis=1),
                "velocity error",
                "turbo",
                "log10",
            )
        )
    if "pressure_error" in fields:
        scalar_specs.append(
            (
                "final_pressure_log_error",
                np.abs(np.asarray(fields["pressure_error"], dtype=float)),
                "pressure error",
                "turbo",
                "log10",
            )
        )

    for stem, values, title, cmap, transform in scalar_specs:
        field_grid = interpolated_scalar_grid(mesh.coords, values, pore, domain_shape, grid_size, transform)
        out = out_dir / f"{stem}.png"
        render_single_panel(title, field_grid, geometry, domain_shape, out, cmap, transform)
        outputs[stem] = str(out)

    if "final_velocity_log_error" not in outputs:
        outputs["final_velocity_log_error"] = None
    if "final_pressure_log_error" not in outputs:
        outputs["final_pressure_log_error"] = None

    overview_inputs = [Path(path) for path in outputs.values() if path]
    if overview_inputs:
        overview = out_dir / "final_state_overview.png"
        combine_panels(overview_inputs, overview)
        outputs["final_state_overview"] = str(overview)
    else:
        outputs["final_state_overview"] = None
    return outputs


def pore_mask(mesh) -> np.ndarray:
    pore = np.zeros(mesh.domain_shape, dtype=bool)
    pore[tuple(mesh.pore_voxels.T)] = True
    return pore


def interpolated_scalar_grid(
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    pore: np.ndarray,
    domain_shape: np.ndarray,
    n: int,
    transform: str,
) -> pv.ImageData:
    sx, sy, sz = domain_shape
    spacing = (sx / (n - 1), sy / (n - 1), sz / (n - 1))
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    points = image.points
    values = idw(points, sample_points, sample_values, k=12)
    volume = values.reshape((n, n, n), order="F")
    fluid = fluid_mask(points, pore).reshape((n, n, n), order="F")
    volume[~fluid] = np.nan

    valid = np.isfinite(volume)
    filled = np.where(valid, volume, 0.0)
    smooth = gaussian_filter(filled, sigma=1.05)
    weights = gaussian_filter(valid.astype(float), sigma=1.05)
    smooth = np.divide(smooth, np.maximum(weights, 1.0e-12))
    smooth[~fluid] = np.nan

    if transform == "log10":
        finite = smooth[np.isfinite(smooth)]
        positive = finite[finite > 0.0]
        floor = max(float(np.percentile(positive, 1.0)) * 0.1, 1.0e-16) if positive.size else 1.0e-16
        smooth = np.log10(np.maximum(smooth, floor))
        scalar_name = "log_error"
    else:
        scalar_name = "value"

    image.point_data[scalar_name] = smooth.reshape(-1, order="F")
    image.point_data["fluid"] = fluid.reshape(-1, order="F").astype(float)
    return image


def build_reference_style_geometry(
    pore: np.ndarray,
    voxel_size: float,
    domain_shape: np.ndarray,
    n: int,
) -> tuple[str, object, object | None]:
    solid = ~pore
    labels, count = ndi.label(solid)
    component_sizes = np.bincount(labels.ravel())[1:]
    largest_component = int(component_sizes.max()) if component_sizes.size else 0
    centers = []
    radii = []
    for label in range(1, count + 1):
        voxels = np.argwhere(labels == label)
        if len(voxels) == 0:
            continue
        center = (voxels.mean(axis=0) + 0.5) * voxel_size
        distances = np.linalg.norm((voxels + 0.5) * voxel_size - center, axis=1)
        radius = float(np.percentile(distances, 82.0)) if len(distances) > 3 else 0.55 * voxel_size
        centers.append(np.clip(center, 0.0, domain_shape))
        radii.append(max(radius, 0.45 * voxel_size))
    if centers and largest_component < 0.25 * solid.size:
        return ("spheres", np.asarray(centers, dtype=float), np.asarray(radii, dtype=float))
    return ("surface", build_solid_surface(pore, domain_shape, n), None)


def build_solid_surface(pore: np.ndarray, domain_shape: np.ndarray, n: int) -> pv.PolyData:
    sx, sy, sz = domain_shape
    spacing = (sx / (n - 1), sy / (n - 1), sz / (n - 1))
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    solid = (~fluid_mask(image.points, pore)).astype(float).reshape((n, n, n), order="F")
    solid = gaussian_filter(solid, sigma=0.95)
    image.point_data["solid"] = solid.reshape(-1, order="F")
    return image.contour(isosurfaces=[0.5], scalars="solid").smooth(
        n_iter=60,
        relaxation_factor=0.08,
        feature_smoothing=False,
        boundary_smoothing=True,
    )


def render_single_panel(
    title: str,
    field_grid: pv.ImageData,
    geometry: tuple[str, object, object],
    domain_shape: np.ndarray,
    out: Path,
    cmap: str,
    transform: str,
) -> None:
    scalar = "log_error" if transform == "log10" else "value"
    values = np.asarray(field_grid.point_data[scalar], dtype=float)
    value_range = finite_range(values)
    contour = field_grid.contour(isosurfaces=iso_levels(value_range), scalars=scalar)

    plotter = pv.Plotter(off_screen=True, window_size=(1080, 860), border=False)
    plotter.set_background("white")
    add_domain(plotter, domain_shape)
    add_reference_geometry(plotter, geometry)
    if contour.n_points:
        plotter.add_mesh(
            contour,
            scalars=scalar,
            cmap=cmap,
            clim=value_range,
            opacity=0.70,
            smooth_shading=True,
            show_edges=False,
            scalar_bar_args={
                "title": "log10 error" if transform == "log10" else title,
                "vertical": True,
                "position_x": 0.87,
                "position_y": 0.18,
                "width": 0.045,
                "height": 0.60,
                "title_font_size": 15,
                "label_font_size": 12,
                "fmt": "%.1f" if transform == "log10" else "%.2g",
                "color": "black",
            },
        )
    plotter.add_text(title, position=(0.035, 0.925), font_size=13, color="black", viewport=True)
    set_camera(plotter, domain_shape)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def render_final_region_methods(mesh, levels: np.ndarray, dof_data: dict[str, np.ndarray], out: Path) -> None:
    labels = final_stage_labels(levels, dof_data)
    voxel_labels = mesh.voxel_labels[tuple(mesh.pore_voxels.T)]
    centers = (mesh.pore_voxels.astype(float) + 0.5) * mesh.voxel_size
    domain_shape = np.asarray(mesh.domain_shape, dtype=float) * mesh.voxel_size

    plotter = pv.Plotter(off_screen=True, window_size=(1080, 860), border=False)
    plotter.set_background("white")
    add_domain(plotter, domain_shape)
    geom = pv.Cube(x_length=0.78 * mesh.voxel_size, y_length=0.78 * mesh.voxel_size, z_length=0.78 * mesh.voxel_size)
    for level, (label, color) in labels.items():
        region_ids = np.flatnonzero(levels == level)
        if not len(region_ids):
            continue
        mask = np.isin(voxel_labels, region_ids)
        cloud = pv.PolyData(centers[mask])
        plotter.add_mesh(cloud.glyph(scale=False, orient=False, geom=geom), color=color, opacity=0.62, label=label)
    plotter.add_text("region methods", position=(0.035, 0.925), font_size=13, color="black", viewport=True)
    plotter.add_legend(size=(0.25, 0.20), bcolor="white", face=None)
    set_camera(plotter, domain_shape)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def final_stage_labels(levels: np.ndarray, dof_data: dict[str, np.ndarray]) -> dict[int, tuple[str, str]]:
    if int(np.max(levels)) > METHOD_LEVEL["HODDPNM"]:
        fractions = np.asarray(dof_data.get("interface_fractions", []), dtype=float)
        hoddpnm_fractions = fractions[fractions > 0.0]
        labels: dict[int, tuple[str, str]] = {
            0: ("PNM", "#5c7cfa"),
            1: ("DDPNM", "#37b24d"),
            2: ("DDPNMT", "#f59f00"),
        }
        palette = ["#ff922b", "#e03131", "#9c36b5", "#364fc7", "#0b7285"]
        for offset, fraction in enumerate(hoddpnm_fractions, start=3):
            labels[offset] = (f"HODDPNM{int(round(100.0 * fraction))}", palette[(offset - 3) % len(palette)])
        return labels
    return {METHOD_LEVEL[method]: (method, METHOD_COLOR[method]) for method in METHOD_COLOR}


def idw(query_points: np.ndarray, sample_points: np.ndarray, sample_values: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(sample_points)
    distances, indices = tree.query(query_points, k=min(k, len(sample_points)), workers=-1)
    weights = 1.0 / np.maximum(distances, 1.0e-8) ** 2
    return np.sum(weights * sample_values[indices], axis=1) / np.sum(weights, axis=1)


def fluid_mask(points: np.ndarray, pore: np.ndarray) -> np.ndarray:
    ijk = np.floor(points).astype(int)
    ijk = np.clip(ijk, 0, np.asarray(pore.shape) - 1)
    return pore[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 4.0))
    hi = float(np.percentile(finite, 99.6))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def iso_levels(value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    return np.linspace(lo + 0.16 * (hi - lo), hi - 0.05 * (hi - lo), 7)


def add_domain(plotter: pv.Plotter, domain_shape: np.ndarray) -> None:
    sx, sy, sz = domain_shape
    cube = pv.Cube(bounds=(0, sx, 0, sy, 0, sz)).extract_surface()
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.08, show_edges=False)


def add_reference_geometry(plotter: pv.Plotter, geometry: tuple[str, object, object | None]) -> None:
    kind, primary, secondary = geometry
    if kind == "spheres":
        add_hole_spheres(plotter, primary, secondary)
        return
    plotter.add_mesh(primary, color="#787878", opacity=0.55, smooth_shading=True, show_edges=False)


def add_hole_spheres(plotter: pv.Plotter, centers: np.ndarray, radii: np.ndarray) -> None:
    for center, radius in zip(centers, radii):
        sphere = make_sphere_polydata(center, radius, theta_resolution=48, phi_resolution=24)
        plotter.add_mesh(sphere, color="#787878", opacity=0.55, smooth_shading=True, show_edges=False)


def make_sphere_polydata(
    center: np.ndarray,
    radius: float,
    theta_resolution: int,
    phi_resolution: int,
) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_resolution, endpoint=False)
    phi = np.linspace(0.0, np.pi, phi_resolution)
    points = []
    for p in phi:
        sin_p = np.sin(p)
        for t in theta:
            points.append(
                [
                    center[0] + radius * sin_p * np.cos(t),
                    center[1] + radius * sin_p * np.sin(t),
                    center[2] + radius * np.cos(p),
                ]
            )
    faces = []
    for i in range(phi_resolution - 1):
        for j in range(theta_resolution):
            a = i * theta_resolution + j
            b = i * theta_resolution + (j + 1) % theta_resolution
            c = (i + 1) * theta_resolution + (j + 1) % theta_resolution
            d = (i + 1) * theta_resolution + j
            faces.extend([4, a, b, c, d])
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))


def set_camera(plotter: pv.Plotter, domain_shape: np.ndarray) -> None:
    center = tuple(0.5 * domain_shape)
    scale = float(max(domain_shape))
    plotter.camera_position = [
        (1.64 * scale, -1.60 * scale, 1.24 * scale),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.90)
    plotter.add_light(pv.Light(position=(0, -0.80 * scale, 1.60 * scale), intensity=0.55))
    plotter.add_light(pv.Light(position=(-0.80 * scale, 0.60 * scale, 1.00 * scale), intensity=0.35))


def combine_panels(paths: list[Path], out: Path) -> None:
    images = [plt.imread(path) for path in paths]
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.006, hspace=0.006)
    fig.savefig(out, dpi=220)
    plt.close(fig)
