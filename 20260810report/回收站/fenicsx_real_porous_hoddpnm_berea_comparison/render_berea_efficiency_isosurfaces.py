from __future__ import annotations

import argparse
import json
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)

ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "outputs" / "02_efficiency_trials" / "aggressive_t025_none_scaled"
DATA_FILE = ROOT / "data" / "berea_100_to_300.npz"
OUT_DIR = ROOT / "outputs" / "03_figures_for_presentation" / "efficiency_aggressive_t025_none_scaled"
CROP = (slice(20, 36), slice(150, 166), slice(30, 46))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--crop", default="20:36,150:166,30:46")
    parser.add_argument("--grid", type=int, default=104)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop = parse_crop(args.crop)
    case_summary = json.loads((args.case_dir / "validation_summary.json").read_text(encoding="utf-8"))
    vtu = pv.read(args.case_dir / "real_porous_hoddpnm_solution.vtu")
    pore = np.load(args.data_file)["im"][crop].astype(bool)
    pore = keep_largest_component(pore)
    active = np.argwhere(pore)
    domain_shape = active.max(axis=0).astype(float) + 1.0

    velocity_error = np.linalg.norm(np.asarray(vtu.point_data["velocity_error"], dtype=float), axis=1)
    pressure_error = np.abs(np.asarray(vtu.point_data["pressure_error"], dtype=float))
    velocity_grid = build_log_error_grid(vtu.points, velocity_error, pore, domain_shape, n=args.grid, sigma=1.05)
    pressure_grid = build_log_error_grid(vtu.points, pressure_error, pore, domain_shape, n=args.grid, sigma=1.05)
    solid_surface = build_solid_surface(pore, domain_shape, n=args.grid)

    velocity_png = args.out_dir / "berea_velocity_log_error_isosurfaces.png"
    pressure_png = args.out_dir / "berea_pressure_log_error_isosurfaces.png"
    combined_png = args.out_dir / "berea_velocity_pressure_log_error_isosurfaces.png"
    velocity_volume_png = args.out_dir / "berea_velocity_log_error_volume.png"
    pressure_volume_png = args.out_dir / "berea_pressure_log_error_volume.png"
    combined_volume_png = args.out_dir / "berea_velocity_pressure_log_error_volume.png"
    render_panel(velocity_grid, solid_surface, domain_shape, "velocity error", velocity_png)
    render_panel(pressure_grid, solid_surface, domain_shape, "pressure error", pressure_png)
    render_volume_panel(velocity_grid, solid_surface, domain_shape, "velocity error", velocity_volume_png)
    render_volume_panel(pressure_grid, solid_surface, domain_shape, "pressure error", pressure_volume_png)
    combine_images(velocity_png, pressure_png, combined_png)
    combine_images(velocity_volume_png, pressure_volume_png, combined_volume_png)

    timings = case_summary.get("timings_seconds", {})
    memory = case_summary.get("memory_trace_mib", {})
    fem_memory = memory.get("after_fem_sparse_direct_solve", {})
    hodd_memory = memory.get("after_hoddpnm_schur_solve") or memory.get("after_hoddpnm_exact_schur_solve", {})
    hodd_time = timings.get("hoddpnm_schur_solve", timings.get("hoddpnm_exact_schur_solve"))
    time_note = {
        "status": "solver-internal timing loaded from validation_summary.json",
        "fem_sparse_direct_solve_seconds": timings.get("fem_sparse_direct_solve"),
        "hoddpnm_schur_solve_seconds": hodd_time,
        "fem_assembly_seconds": timings.get("fem_assembly"),
        "geometry_and_decomposition_seconds": timings.get("geometry_and_decomposition"),
        "total_before_output_seconds": timings.get("total_before_output"),
        "fem_checkpoint_working_set_mib": fem_memory.get("working_set_mib"),
        "fem_checkpoint_peak_working_set_mib": fem_memory.get("peak_working_set_mib"),
        "hoddpnm_checkpoint_working_set_mib": hodd_memory.get("working_set_mib"),
        "hoddpnm_checkpoint_peak_working_set_mib": hodd_memory.get("peak_working_set_mib"),
    }
    summary = {
        "case": "real Berea sandstone crop",
        "crop": f"x={crop[0].start}:{crop[0].stop}, y={crop[1].start}:{crop[1].stop}, z={crop[2].start}:{crop[2].stop} from OpenPNM berea_100_to_300.npz",
        "style_source": r"D:\hu\tongjiproj\FENICSX\fenicsx_irregular_hoddpnm\outputs\cube_holes_27_taylor_hood_random_style_errors\cells_9_pressure_error_isosurface.png",
        "visualization": "IDW interpolation to a tight regular grid, Gaussian smoothing, solid voxels masked from error field, solid rock rendered as a grey translucent isosurface, random-style white-background panels",
        "grid": args.grid,
        "display_domain_shape": domain_shape.tolist(),
        "velocity_log10_range": finite_range(velocity_grid.point_data["log_error"]),
        "pressure_log10_range": finite_range(pressure_grid.point_data["log_error"]),
        "case_summary": case_summary,
        "time_comparison": time_note,
        "outputs": {
            "velocity": str(velocity_png),
            "pressure": str(pressure_png),
            "combined": str(combined_png),
            "velocity_volume": str(velocity_volume_png),
            "pressure_volume": str(pressure_volume_png),
            "combined_volume": str(combined_volume_png),
        },
    }
    (args.out_dir / "berea_paper_isosurface_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["time_comparison"], indent=2))
    print(f"wrote {combined_png}")


def parse_crop(text: str) -> tuple[slice, slice, slice]:
    spans = []
    for part in text.split(","):
        start, stop = part.split(":")
        spans.append(slice(int(start), int(stop)))
    if len(spans) != 3:
        raise SystemExit("--crop must be x0:x1,y0:y1,z0:z1")
    return tuple(spans)


def build_log_error_grid(
    sample_points: np.ndarray,
    sample_error: np.ndarray,
    pore: np.ndarray,
    domain_shape: np.ndarray,
    n: int,
    sigma: float,
) -> pv.ImageData:
    sx, sy, sz = domain_shape
    spacing = (sx / (n - 1), sy / (n - 1), sz / (n - 1))
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    points = image.points
    values = idw(points, sample_points, sample_error, k=12)
    volume = values.reshape((n, n, n), order="F")
    fluid = fluid_mask(points, pore).reshape((n, n, n), order="F")
    volume[~fluid] = np.nan

    valid = np.isfinite(volume)
    filled = np.where(valid, volume, 0.0)
    smooth = gaussian_filter(filled, sigma=sigma)
    weights = gaussian_filter(valid.astype(float), sigma=sigma)
    smooth = np.divide(smooth, np.maximum(weights, 1.0e-12))
    smooth[~fluid] = np.nan

    finite = smooth[np.isfinite(smooth)]
    floor = max(float(np.percentile(finite, 2.0)) * 0.1, 1.0e-16)
    image.point_data["log_error"] = np.log10(np.maximum(smooth.reshape(-1, order="F"), floor))
    image.point_data["fluid"] = fluid.reshape(-1, order="F").astype(float)
    return image


def build_solid_surface(pore: np.ndarray, domain_shape: np.ndarray, n: int) -> pv.PolyData:
    sx, sy, sz = domain_shape
    spacing = (sx / (n - 1), sy / (n - 1), sz / (n - 1))
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    points = image.points
    solid = (~fluid_mask(points, pore)).astype(float).reshape((n, n, n), order="F")
    solid = gaussian_filter(solid, sigma=1.0)
    image.point_data["solid"] = solid.reshape(-1, order="F")
    surface = image.contour(isosurfaces=[0.5], scalars="solid")
    return surface.smooth(n_iter=80, relaxation_factor=0.08, feature_smoothing=False, boundary_smoothing=True)


def render_panel(field: pv.ImageData, solid_surface: pv.PolyData, domain_shape: np.ndarray, title: str, out: Path) -> None:
    clim = finite_range(field.point_data["log_error"])
    contour = field.contour(isosurfaces=iso_levels(clim), scalars="log_error")
    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(off_screen=True, window_size=(1080, 860), border=False)
    plotter.set_background("white")
    add_domain_box(plotter, domain_shape)
    plotter.add_mesh(
        solid_surface,
        color="#787878",
        opacity=0.55,
        smooth_shading=True,
        show_edges=False,
    )
    plotter.add_mesh(
        contour,
        scalars="log_error",
        cmap="turbo",
        clim=clim,
        opacity=0.70,
        smooth_shading=True,
        show_edges=False,
        scalar_bar_args={
            "title": "log10 error",
            "vertical": True,
            "position_x": 0.87,
            "position_y": 0.18,
            "width": 0.045,
            "height": 0.60,
            "title_font_size": 15,
            "label_font_size": 12,
            "fmt": "%.1f",
            "color": "black",
        },
    )
    plotter.add_text(title, position=(0.035, 0.925), font_size=13, color="black", viewport=True)
    set_camera(plotter, domain_shape)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def render_volume_panel(field: pv.ImageData, solid_surface: pv.PolyData, domain_shape: np.ndarray, title: str, out: Path) -> None:
    clim = finite_range(field.point_data["log_error"])
    volume_field = field.copy()
    volume_values = np.asarray(volume_field.point_data["log_error"], dtype=float).copy()
    volume_values[~np.isfinite(volume_values)] = clim[0]
    volume_field.point_data["volume_log_error"] = volume_values
    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(off_screen=True, window_size=(1080, 860), border=False)
    plotter.set_background("white")
    add_domain_box(plotter, domain_shape)
    plotter.add_mesh(solid_surface, color="#787878", opacity=0.32, smooth_shading=True, show_edges=False)
    plotter.add_volume(
        volume_field,
        scalars="volume_log_error",
        cmap="turbo",
        clim=clim,
        opacity=[0.0, 0.0, 0.02, 0.07, 0.17, 0.38, 0.76],
        shade=True,
        mapper="smart",
        scalar_bar_args={
            "title": "log10 error",
            "vertical": True,
            "position_x": 0.87,
            "position_y": 0.18,
            "width": 0.045,
            "height": 0.60,
            "title_font_size": 15,
            "label_font_size": 12,
            "fmt": "%.1f",
            "color": "black",
        },
    )
    plotter.add_text(title, position=(0.035, 0.925), font_size=13, color="black", viewport=True)
    set_camera(plotter, domain_shape)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def add_domain_box(plotter: pv.Plotter, domain_shape: np.ndarray) -> None:
    sx, sy, sz = domain_shape
    cube = pv.Cube(bounds=(0, sx, 0, sy, 0, sz)).extract_surface()
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.08, show_edges=False)


def set_camera(plotter: pv.Plotter, domain_shape: np.ndarray) -> None:
    sx, sy, sz = domain_shape
    center = (0.5 * sx, 0.5 * sy, 0.5 * sz)
    scale = float(max(domain_shape))
    plotter.camera_position = [
        (4.10 * scale, -4.00 * scale, 3.10 * scale),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(2.25)
    plotter.add_light(pv.Light(position=(0, -2.0 * scale, 4.0 * scale), intensity=0.55))
    plotter.add_light(pv.Light(position=(-2.0 * scale, 1.5 * scale, 2.5 * scale), intensity=0.35))


def idw(query_points: np.ndarray, sample_points: np.ndarray, sample_values: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(sample_points)
    distances, indices = tree.query(query_points, k=min(k, len(sample_points)), workers=-1)
    weights = 1.0 / np.maximum(distances, 1.0e-8) ** 2
    return np.sum(weights * sample_values[indices], axis=1) / np.sum(weights, axis=1)


def fluid_mask(points: np.ndarray, pore: np.ndarray) -> np.ndarray:
    ijk = np.floor(points).astype(int)
    ijk = np.clip(ijk, 0, np.asarray(pore.shape) - 1)
    return pore[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def keep_largest_component(pore: np.ndarray) -> np.ndarray:
    labels = -np.ones(pore.shape, dtype=np.int32)
    best_label = -1
    best_size = 0
    current = 0
    for start in np.argwhere(pore):
        sx, sy, sz = (int(v) for v in start)
        if labels[sx, sy, sz] >= 0:
            continue
        q = deque([(sx, sy, sz)])
        labels[sx, sy, sz] = current
        size = 0
        while q:
            x, y, z = q.popleft()
            size += 1
            for nx, ny, nz in neighbors(x, y, z, pore.shape):
                if pore[nx, ny, nz] and labels[nx, ny, nz] < 0:
                    labels[nx, ny, nz] = current
                    q.append((nx, ny, nz))
        if size > best_size:
            best_size = size
            best_label = current
        current += 1
    return labels == best_label


def neighbors(x: int, y: int, z: int, shape: tuple[int, int, int]):
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            yield nx, ny, nz


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def iso_levels(value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    return np.linspace(lo + 0.16 * (hi - lo), hi - 0.05 * (hi - lo), 7)


def combine_images(left: Path, right: Path, out: Path) -> None:
    left_img = Image.open(left).convert("RGB")
    right_img = Image.open(right).convert("RGB")
    height = min(left_img.height, right_img.height)
    left_img = resize_to_height(left_img, height)
    right_img = resize_to_height(right_img, height)
    gap = max(8, int(0.015 * height))
    canvas = Image.new("RGB", (left_img.width + gap + right_img.width, height), "white")
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width + gap, 0))
    canvas.save(out)


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = int(round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    main()
