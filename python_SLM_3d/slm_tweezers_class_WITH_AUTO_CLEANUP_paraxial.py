from __future__ import annotations

"""
Stateful, classâ€‘based refactor scaffold for the SLM tweezers code.

Goal: keep functionality identical while turning the script into a single
stateful object with:
  â€¢ a clear initialization phase (loads YAML, sets sizes/dtypes, allocates arrays)
  â€¢ explicit configuration and cached derived values
  â€¢ methods that mutate internal state instead of returning huge tuples

This file is a *dropâ€‘in architecture* you can paste into your repo and then
progressively migrate existing helper logic into the marked methods. I've
included conservative, working implementations for the plumbing and left
wellâ€‘labeled TODO hooks where you can paste/port your existing math (GS loop,
masking variants, etc.). The goal is to avoid another round of undefined names
by providing all attributes in one place.

USAGE
-----
slm = SLMTweezers(yaml_path="../slm_parameters.yml", redSLM=1, scal=1)
slm.init_fields(waist_um=9/2*1e3)
slm.set_target_grid(n_horiz=3, n_vert=3, spacing_um=7.2, odd_tw=1)
slm.build_tweezers_box(box1=1)
slm.run_gs(iterations=30, Gg=0.6, useFilter=False)
slm.save_pickle(label="_3x3_7p2um")

You can now call followâ€‘ups like `slm.set_target_grid(...)` and `slm.run_gs(...)`
without reâ€‘loading YAML or reâ€‘allocating base arrays.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import datetime
import pickle
import time
import numpy as np
import yaml
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import math
import gc

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # type: ignore


# ============================================================================
# MEMORY MANAGEMENT UTILITIES (Auto-cleanup for PyCharm performance)
# ============================================================================

def _cleanup_memory(verbose: bool = False):
    """
    Automatically clean up GPU and CPU memory after GS operations.
    This prevents PyCharm slowdown by releasing GPU/CPU memory.
    """
    if verbose:
        print("  🧹 Cleaning up memory...")

    # Force Python garbage collection
    gc.collect()

    # Try to clear PyTorch cache (MPS on Mac, CUDA on GPU)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, Exception):
        pass

    # Try to clear CuPy memory pool
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except (ImportError, Exception):
        pass

    # Final garbage collection
    gc.collect()

    if verbose:
        print("  ✓ Memory cleanup complete")



@dataclass(slots=True)
class SLMParams:
    x_pixels: int
    y_pixels: int
    pixel_um: float

    # Optional fields from your YAML we ignore for math
    bit_depth: int | None = None
    lut: str | None = None
    wfc: str | None = None
    RGB: bool | None = None
    bytesPerPixel: int | None = None
    x_size_mm: float | None = None
    y_size_mm: float | None = None


@dataclass(slots=True)
class GSConfig:
    redSLM: int = 1
    scal: int | float = 1
    odd_tw: int = 1  # 1 = centered grid path
    useFilter: bool = False
    angle_adjust: float = 0.0  # ignored (rotation removed)
    quadrantspacing_horiz: int = 0
    quadrantspacing_vert: int = 0
    array_offset_top: int = 0
    pos_adjust: int = 0


@dataclass(slots=True)
class ResultBundle:
    phase_mask: np.ndarray
    A_in: np.ndarray
    tweezlist: np.ndarray
    weights: np.ndarray
    tweezer_mask: np.ndarray
    convergence: int
    file: str


# --- Backend selector: CuPy (GPU) > PyTorch (GPU/MPS) > NumPy (CPU) ---
# --- Backend selector: CuPy (GPU) > PyTorch (CUDA/MPS) > NumPy (CPU) ---
class _FFTBackend:
    def __init__(self, prefer: str | None = None):
        self.name = "numpy"
        self.xp = None
        self._fft2 = self._ifft2 = self._fftshift = self._ifftshift = None
        self._asarray = self._to_numpy = None
        # Expose common dtypes
        self.complex64 = None
        self.float32 = None
        self.int64 = None

        tried = []

        # --- Try CuPy first ---
        if prefer in (None, "cupy"):
            try:
                print("[Backend] Attempting to import CuPy...")
                import cupy as cp
                print("[Backend] CuPy imported successfully")

                print("[Backend] Checking CUDA availability...")
                if cp.cuda.is_available():
                    device = cp.cuda.Device()
                    print(f"[Backend] [OK] CUDA available - GPU: {device.compute_capability}")
                else:
                    print("[Backend] [WARN] CUDA not available")

                print("[Backend] Importing CuPy FFT functions...")
                from cupyx.scipy.fft import fft2, ifft2, fftshift, ifftshift
                print("[Backend] FFT functions imported")
                self.name = "cupy"
                self.xp = cp
                self._fft2, self._ifft2 = fft2, ifft2
                self._fftshift, self._ifftshift = fftshift, ifftshift
                self._asarray = cp.asarray
                self._to_numpy = cp.asnumpy
                self.complex64 = cp.complex64
                self.float32 = cp.float32
                self.int64 = cp.int64

                # e^{iÎ¸} helper
                self.exp_i = lambda theta: cp.exp(1j * theta)
                print("[Backend] [OK] CuPy backend initialized successfully")
                return
            except Exception as e:
                print(f"[Backend] [FAIL] CuPy initialization failed: {e}")
                tried.append(("cupy", str(e)))

        # --- Try PyTorch next (CUDA > MPS > CPU) ---
        # --- Try PyTorch next (CUDA > MPS > CPU) ---
        if prefer in (None, "torch"):
            try:
                print("[Backend] Attempting to import PyTorch...")
                import torch
                import torch.fft as tfft
                print("[Backend] PyTorch imported successfully")

                print("[Backend] Detecting PyTorch device...")
                if torch.cuda.is_available():
                    _device = torch.device("cuda")
                    print(f"[Backend] [OK] Using CUDA device")
                elif torch.backends.mps.is_available():
                    _device = torch.device("mps")
                    print(f"[Backend] [OK] Using MPS device (Apple Silicon)")
                else:
                    _device = torch.device("cpu")
                    print(f"[Backend] Using CPU device")

                class TorchXP:
                    # filled after class definition
                    device = None
                    complex64 = torch.complex64
                    float32 = torch.float32
                    int64 = torch.int64

                    @staticmethod
                    def asarray(a, dtype=None):
                        t = torch.as_tensor(a, device=TorchXP.device)
                        if dtype is not None and t.dtype != dtype:
                            t = t.to(dtype)
                        return t

                    @staticmethod
                    def zeros_like(a, dtype=None):
                        return torch.zeros_like(a, dtype=(dtype or a.dtype), device=a.device)

                    @staticmethod
                    def abs(a):      return torch.abs(a)

                    @staticmethod
                    def angle(a):    return torch.angle(a)

                    @staticmethod
                    def reshape(a, shape): return a.reshape(*shape)

                    @staticmethod
                    def ravel(a):    return a.reshape(-1)

                    @staticmethod
                    def mean(a):
                        import torch
                        return torch.mean(a)

                # bind device AFTER class is created
                TorchXP.device = _device

                def fftshift(x):
                    return tfft.fftshift(x, dim=(-2, -1))

                def ifftshift(x):
                    return tfft.ifftshift(x, dim=(-2, -1))

                self.name = f"torch[{_device.type}]"
                self.xp = TorchXP
                self._fft2 = lambda a: tfft.fft2(a)
                self._ifft2 = lambda a: tfft.ifft2(a)
                self._fftshift = fftshift
                self._ifftshift = ifftshift
                self._asarray = TorchXP.asarray
                self._to_numpy = lambda a: a.detach().cpu().numpy()
                self.complex64 = torch.complex64
                self.float32 = torch.float32
                self.int64 = torch.int64

                # robust e^{iÎ¸}
                def exp_i(theta):
                    mag = torch.ones_like(theta, device=theta.device, dtype=theta.dtype)
                    return torch.polar(mag, theta)  # complex result

                self.exp_i = exp_i

                print(f"[Backend] [OK] PyTorch backend initialized successfully on {_device.type}")
                return
            except Exception as e:
                print(f"[Backend] [FAIL] PyTorch initialization failed: {e}")
                tried.append(("torch", str(e)))

        # --- Fallback: NumPy ---
        print("[Backend] Falling back to NumPy (CPU-only)...")
        import numpy as _np
        from numpy.fft import fft2, ifft2, fftshift, ifftshift
        self.name = "numpy"
        self.xp = _np
        self._fft2, self._ifft2 = fft2, ifft2
        self._fftshift, self._ifftshift = fftshift, ifftshift
        self._asarray = lambda a, dtype=None: _np.asarray(a, dtype=dtype)
        self._to_numpy = lambda a: a
        self.complex64 = _np.complex64
        self.float32 = _np.float32
        self.int64 = _np.int64
        self.exp_i = lambda theta: _np.exp(1j * theta)

        if tried:
            print("[Backend] Falling back to NumPy. Tried and failed:", tried)

    # thin wrappers
    def asarray(self, a, dtype=None):
        return self._asarray(a, dtype=dtype)

    def fft2(self, a):
        return self._fft2(a)

    def ifft2(self, a):
        return self._ifft2(a)

    def fftshift(self, a):
        return self._fftshift(a)

    def ifftshift(self, a):
        return self._ifftshift(a)

    def to_numpy(self, a):
        return self._to_numpy(a)


class SLMTweezers:
    def __init__(self, yaml_path: str | Path, *, redSLM: int = 1, scal: int | float = 1) -> None:
        # Backend (CPU/GPU)
        self.backend = _FFTBackend(prefer=None)  # None | "cupy" | "torch"
        if hasattr(self.backend, "name"):
            print(f"[Backend] Using {self.backend.name} for FFTs")

        # Resolve yaml_path robustly:
        # - If an absolute path is provided, use it.
        # - If a relative path is provided, prefer resolving it relative to
        #   this source file (so debugger/run working directory differences
        #   don't break loading). If that doesn't exist, fall back to the
        #   path as given (so callers can still pass cwd-relative paths).
        given = Path(yaml_path)
        if given.is_absolute():
            self.yaml_path = given
        else:
            # Candidate relative to this script's directory
            candidate = (Path(__file__).resolve().parent / given).resolve()
            if candidate.exists():
                self.yaml_path = candidate
            else:
                # Fall back to given path (relative to current working dir)
                self.yaml_path = given
        self.config = GSConfig(redSLM=redSLM, scal=scal)

        self.params = self._load_yaml(self.yaml_path)
        self.rng = np.random.default_rng()

        # Derived sizes (after redSLM downsampling)
        self.x_pixels1 = self.params.x_pixels // self.config.redSLM
        self.y_pixels1 = self.params.y_pixels // self.config.redSLM

        # Placeholders for arrays/state (filled by init_fields / set_target_grid)
        self.A_in: Optional[np.ndarray] = None
        self.pad: Optional[np.ndarray] = None
        self.psi0: Optional[np.ndarray] = None
        self.A_target: Optional[np.ndarray] = None
        self.center_row: Optional[int] = None
        self.center_col: Optional[int] = None
        self.filter_2d: Optional[np.ndarray] = None
        self.box1: int = 1
        self.coordinates: Optional[np.ndarray] = None
        self.height_corr: Optional[np.ndarray] = None
        self.height_corr2: Optional[np.ndarray] = None
        self.target_xy_um: Optional[np.ndarray] = None

        # Output cache
        self.phase_mask: Optional[np.ndarray] = None
        self.tweezlist: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.tweezer_mask: Optional[np.ndarray] = None
        self.convergence: int = 0
        self.last_file: Optional[str] = None

        # Optics (set these once)
        self.wavelength_um: float | None = None
        self.focal_length_um: float | None = None

        # Pupil coordinate caches (Âµm^2)
        self._pupil_X2: Optional[np.ndarray] = None
        self._pupil_Y2: Optional[np.ndarray] = None
        self._pupil_XY: Optional[np.ndarray] = None
        self._k_over_2f2: Optional[float] = None

        # Per-spot quadratic coefficients (defocus/astig), one per tweezer
        #  phase_i(X,Y) = (k/2f^2) * (alpha_i * X^2 + beta_i * Y^2 + gamma_i * X*Y)
        self.defocus_alpha: Optional[np.ndarray] = None
        self.defocus_beta: Optional[np.ndarray] = None
        self.defocus_gamma: Optional[np.ndarray] = None  # optional cross-term

    def set_optics(self, *, wavelength_um: float, focal_length_um: float) -> None:
        """
        Define the imaging optics (SLM pupil -> focal plane).
        Also precompute pupil coordinate grids X^2, Y^2, X*Y (Âµm^2).
        """
        self.wavelength_um = float(wavelength_um)
        self.focal_length_um = float(focal_length_um)

        # Build pupil coordinate grids on the *A_target* canvas (pad size)
        assert self.A_target is not None, "Call set_target_grid() before set_optics()."
        H, W = self.A_target.shape
        pu = float(self.params.pixel_um)
        x = (np.arange(W) - W / 2.0) * pu
        y = (np.arange(H) - H / 2.0) * pu
        XX, YY = np.meshgrid(x, y)
        self._pupil_X2 = (XX * XX).astype(np.float32)
        self._pupil_Y2 = (YY * YY).astype(np.float32)
        self._pupil_XY = (XX * YY).astype(np.float32)

        k = 2.0 * np.pi / self.wavelength_um
        self._k = float(k)  # Store k for exact formula
        self._k_over_2f2 = float(k / (2.0 * self.focal_length_um ** 2))  # Keep for compatibility

    def set_tilt_plane(self, *, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0) -> None:
        """
        Specify a *tilted best-focus plane* by giving slopes along X and Y.
        Each tweezer i gets a per-spot defocus z_i = tan(theta_x)*x_i + tan(theta_y)*y_i.
        This is mapped to isotropic quadratics: alpha_i = beta_i = z_i (no astigmatism).
        """
        assert self.target_xy_um is not None, "Call set_target_grid() first."
        tx = np.tan(np.deg2rad(float(tilt_x_deg)))
        ty = np.tan(np.deg2rad(float(tilt_y_deg)))
        xy = self.target_xy_um.astype(np.float64)
        x0 = float(np.mean(xy[:, 0]))
        y0 = float(np.mean(xy[:, 1]))
        z_i = tx * (xy[:, 0] - x0) + ty * (xy[:, 1] - y0)  # Âµm of defocus per spot
        self.defocus_alpha = z_i.astype(np.float32).copy()
        self.defocus_beta = z_i.astype(np.float32).copy()
        self.defocus_gamma = np.zeros_like(self.defocus_alpha, dtype=np.float32)

    def set_quadratic_maps(self, *, alpha: np.ndarray, beta: np.ndarray, gamma: Optional[np.ndarray] = None) -> None:
        """
        Directly set per-spot quadratic coefficients from measurement fits.
        alpha,beta,gamma must be length = number of tweezers (same order as self.tweezlist).
        """
        self.defocus_alpha = np.asarray(alpha, dtype=np.float32)
        self.defocus_beta = np.asarray(beta, dtype=np.float32)
        if gamma is None:
            self.defocus_gamma = np.zeros_like(self.defocus_alpha, dtype=np.float32)
        else:
            self.defocus_gamma = np.asarray(gamma, dtype=np.float32)

    def _phi_quad_for_spot(self, i: int) -> np.ndarray:
        """
        Build the quadratic phase screen for spot i on the pupil canvas.
        Ï†_i(X,Y) = (k/2f^2)*(Î±_i X^2 + Î²_i Y^2 + Î³_i X Y).
        """
        assert self._pupil_X2 is not None and self._pupil_Y2 is not None and self._k_over_2f2 is not None
        ai = float(self.defocus_alpha[i]) if self.defocus_alpha is not None else 0.0
        bi = float(self.defocus_beta[i]) if self.defocus_beta is not None else 0.0
        gi = float(self.defocus_gamma[i]) if self.defocus_gamma is not None else 0.0
        return self._k_over_2f2 * (ai * self._pupil_X2 + bi * self._pupil_Y2 + gi * self._pupil_XY)

    # -------- initialization phase --------
    @staticmethod
    def _load_yaml(path: Path) -> SLMParams:
        if not path.exists():
            raise FileNotFoundError(f"SLM YAML file not found: {path!s}.\n"
                                    f"Tried resolving relative to script directory and CWD.\n"
                                    f"Pass an absolute path or check working directory/launch.json.")

        data = yaml.load(path.read_text(), Loader=Loader) or {}
        sp = data.get("slm_parameters", {})
        return SLMParams(
            x_pixels=int(sp["x_pixels"]),
            y_pixels=int(sp["y_pixels"]),
            pixel_um=float(sp["pixel_um"]),
            bit_depth=sp.get("bit_depth"),
            lut=sp.get("lut"),
            wfc=sp.get("wfc"),
            RGB=sp.get("RGB"),
            bytesPerPixel=sp.get("bytesPerPixel"),
            x_size_mm=sp.get("x_size_mm"),
            y_size_mm=sp.get("y_size_mm"),
        )

    def init_fields(self, *, waist_um: float, dtype=np.float32) -> None:
        """Allocate input Gaussian field on the reduced grid and the rectangular pad.
        Also sets a random initial phase psi0 on the *unreduced* grid.
        """
        xpix, ypix = self.x_pixels1, self.y_pixels1
        scal = int(self.config.scal)

        # Gaussian waist in pixels (reduced grid)
        waist_in = waist_um / self.params.pixel_um
        mid_y_start = int((scal - 1) * ypix / 2)
        mid_y_end = int((scal + 1) * ypix / 2)
        mid_x_start = int((scal - 1) * xpix / 2)
        mid_x_end = int((scal + 1) * xpix / 2)

        # reduced grid open coordinates
        y, x = np.ogrid[1:ypix + 1, 1:xpix + 1]  # y array of size: (ypix,1), x: (1,xpix)

        cy = np.ceil(ypix / 2.0)
        cx = np.ceil(xpix / 2.0)

        core = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (waist_in ** 2))).astype(np.float32)
        A_in = np.zeros((scal * ypix, scal * xpix), dtype=np.float32)
        A_in[mid_y_start:mid_y_end, mid_x_start:mid_x_end] = core
        self.A_in = A_in

        # Rectangular pad on center reduced grid
        H, W = scal * ypix, scal * xpix  # full padded canvas size
        pad = np.zeros((H, W), dtype=np.float32)
        pad[mid_y_start:mid_y_end, mid_x_start:mid_x_end] = 1.0  # ones on the SLM area, zeros elsewhere
        self.pad = pad

        # Initial random phase on reduced grid
        self.psi0 = (2 * np.pi * self.rng.random((ypix, xpix))).astype(np.float32)

    # -------- target grid / tweezers --------
    def _calc_spacing_pixels(self, spacing_um: float, A_single) -> int:
        # QUESTION FOR HANNAH, I'M SO CONFUSED HEREEEEEE, HOW IS SPACING CALCULATED
        # Find the maximum intensity and its index
        max_val = np.max(np.abs(A_single) ** 2)
        max_idx = np.argmax(np.abs(A_single) ** 2)

        # Convert flat index to 2D indices (Python uses different indexing)

        # Find the 1/e^2 radius
        val = max_val

        # This requires modification from MATLAB code since we need to work with 2D indices
        # We'll use a different approach to find the 1/e^2 radius
        # Creating a flattened copy for traversal
        A_single_flat = np.abs(A_single.flatten()) ** 2
        stop_idx = max_idx

        while val >= np.exp(-2) * max_val:  # 1/e^2 radius
            stop_idx += 1
            if stop_idx < len(A_single_flat):
                val = A_single_flat[stop_idx]
            else:
                break

        # Define beam width and spacing parameters
        spacing_factor = 1.14 * spacing_um / (10.2 / 4.1)  # second number is in um
        spacing = int(np.ceil(spacing_factor * 2 * (stop_idx - max_idx)))
        return max(1, spacing)

    def set_target_grid(self, *, n_horiz: int, n_vert: int, spacing_um: float, odd_tw: int = 1,
                        ifcircle: int = 0, circrad: Optional[float] = None,
                        box1: int = 1) -> None:
        """Construct A_target with unit amplitudes at desired tweezer centers.
        Rotation removed: lattice is axisâ€‘aligned. Spacing matches legacy FFT-based
        estimation to preserve behavior.
        """
        assert self.A_in is not None, "Call init_fields() first."
        xpix, ypix = self.x_pixels1, self.y_pixels1

        # Fourier plane of a single input for center finding
        A_single = fftshift(fft2(ifftshift(self.A_in)))
        power = np.abs(A_single) ** 2
        center_idx = np.argmax(power)
        self.center_row, self.center_col = np.unravel_index(center_idx, A_single.shape)

        # spacing in pixels (reduced grid) via legacy mapping
        spacing_h = self._calc_spacing_pixels(spacing_um, A_single)
        spacing_v = spacing_h * ypix / xpix

        # Generate centered tweezers
        h_offset_curr = np.round(spacing_h * (np.arange(n_horiz) - (n_horiz - 1) / 2))
        v_offset_curr = np.round(spacing_v * (np.arange(n_vert) - (n_vert - 1) / 2))

        target_rows_ref = np.repeat(v_offset_curr, n_horiz)
        target_cols_ref = np.tile(h_offset_curr, n_vert)

        target_rows = np.round(self.center_row + target_rows_ref).astype(int)
        target_cols = np.round(self.center_col + target_cols_ref).astype(int)

        # Build target with boundary-safe centers only
        A_target = np.zeros_like(A_single, dtype=np.float32)
        valid = (target_rows >= box1) & (target_rows < A_target.shape[0] - box1) & \
                (target_cols >= box1) & (target_cols < A_target.shape[1] - box1)

        if not np.all(valid):
            print(f"Warning: {(~valid).sum()} tweezer(s) outside scal*xpixels, scal*ypixels w/ box1={box1} boundary")

        A_target[target_rows[valid], target_cols[valid]] = 1.0  # Direct indexing is faster than ravel_multi_index

        self.A_target = A_target
        self.box1 = int(box1)

        # Build per-tweezer pixel blocks and coordinates (exact K*(2b+1)^2)
        self.tweezlist, self.coordinates = self._compute_tweezer_centers_and_coords(A_target, target_rows[valid],
                                                                                    target_cols[valid], self.box1)

        # Height corrections (identity by default)
        self.height_corr = np.ones((len(self.tweezlist), 1), dtype=np.float64) if len(self.tweezlist) else np.empty(
            (0, 1))
        self.height_corr2 = np.repeat(self.height_corr, (2 * self.box1 + 1) ** 2, axis=0)

        # Simple binary mask
        self.tweezer_mask = (A_target > 0).astype(np.uint8)

        # --- NEW: physical target positions in the focal plane (Âµm), centered ---
        x_um_axis = (np.arange(n_horiz) - (n_horiz - 1) / 2.0) * float(spacing_um)
        y_um_axis = (np.arange(n_vert) - (n_vert - 1) / 2.0) * float(spacing_um)

        # Match the same enumeration order you use for pixels: for each y, sweep all x
        x_um_list = np.tile(x_um_axis, n_vert)
        y_um_list = np.repeat(y_um_axis, n_horiz)

        # Keep only spots that survived the in-bounds check
        xy_um = np.stack([x_um_list[valid], y_um_list[valid]], axis=1).astype(np.float32)

        # Center just in case n_horiz/n_vert are even and rounding produced off-by-Â½ issues
        xy_um[:, 0] -= np.mean(xy_um[:, 0])
        xy_um[:, 1] -= np.mean(xy_um[:, 1])

        self.target_xy_um = xy_um  # shape (K, 2): columns [x_um, y_um]

        return 0

    @staticmethod
    def _mask_from_targets(A_target: np.ndarray, n_horiz: int, n_vert: int) -> np.ndarray:
        # Minimal rectangular mask for convenience
        return (A_target > 0).astype(np.uint8)

    @staticmethod
    def _compute_tweezer_centers_and_coords(A_target: np.ndarray, target_rows: np.ndarray, target_cols: np.ndarray,
                                            box1: int) -> tuple[np.ndarray, np.ndarray]:
        centers: list[tuple[int, int]] = []
        H, W = A_target.shape
        for r, c in zip(target_rows.astype(int), target_cols.astype(int)):
            # Ensure full (2b+1)x(2b+1) neighborhood is in-bounds
            if (r - box1) < 0 or (r + box1) >= H or (c - box1) < 0 or (c + box1) >= W:
                continue
            if A_target[r, c] != 0:
                centers.append((int(r), int(c)))
        side = 2 * box1 + 1
        coords: list[int] = []
        for (r0, c0) in centers:
            v = np.arange(r0 - box1, r0 + box1 + 1)
            h = np.arange(c0 - box1, c0 + box1 + 1)
            grid_rows = np.tile(v, (side, 1)).ravel()
            grid_cols = np.repeat(h, side)
            flat_idx = np.ravel_multi_index((grid_rows, grid_cols), A_target.shape)
            coords.extend(flat_idx.tolist())
        tweezlist = np.array(centers, dtype=int) if centers else np.empty((0, 2), dtype=int)
        coordinates = np.array(coords, dtype=int) if coords else np.empty((0,), dtype=int)
        return tweezlist, coordinates

    # -------- GS loop & output --------
    def run_gs(self, *, iterations: int, Gg: float, useFilter: bool = False, tol: float = 1e-4) -> None:
        """
        Classic single-plane GS with early-stop tolerance.
        """
        assert self.A_in is not None and self.pad is not None and self.A_target is not None and self.psi0 is not None
        assert self.coordinates is not None and self.height_corr2 is not None

        if useFilter:
            y = np.arange(self.A_target.shape[0])
            x = np.arange(self.A_target.shape[1])
            X, Y = np.meshgrid(x, y)
            self.filter_2d = 1.0 + np.sqrt((self.center_col - X) ** 2 + (self.center_row - Y) ** 2) / 3000 + Y / 6000
        else:
            self.filter_2d = np.ones_like(self.A_target)

        start_time = time.time()

        psi = self._gs_loop(
            A_in=self.A_in,
            pad=self.pad,
            A_target=self.A_target,
            height_corr2=self.height_corr2,
            psi0=self.psi0,
            iterations=iterations,
            Gg=Gg,
            filter_2d=self.filter_2d,
            coordinates=self.coordinates,
            box1=self.box1,
            tol=tol,  # NEW
        )

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")

        scal = int(self.config.scal)
        ypix, xpix = self.A_in.shape[0] // scal, self.A_in.shape[1] // scal
        mid_y_start = int((scal - 1) * ypix / 2)
        mid_y_end = int((scal + 1) * ypix / 2)
        mid_x_start = int((scal - 1) * xpix / 2)
        mid_x_end = int((scal + 1) * xpix / 2)
        phase_mask = psi[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
        self.phase_mask = np.mod(phase_mask, 2 * np.pi)

        _cleanup_memory(verbose=False)

    def _gs_loop(self, *, A_in: np.ndarray, pad: np.ndarray, A_target: np.ndarray, height_corr2: np.ndarray,
                 psi0: np.ndarray, iterations: int, Gg: float, filter_2d: np.ndarray,
                 coordinates: np.ndarray, box1: int, tol: float) -> np.ndarray:
        """
        Port of the legacy GS loop with per-tweezer weighting and configurable tolerance.
        """
        g = np.ones_like(A_target, dtype=A_target.dtype)
        B_target = np.abs(A_target)

        psi = np.zeros_like(A_target, dtype=np.float32)
        my = (psi.shape[0] - psi0.shape[0]) // 2
        mx = (psi.shape[1] - psi0.shape[1]) // 2
        psi[my:my + psi0.shape[0], mx:mx + psi0.shape[1]] = psi0

        side = 2 * box1 + 1
        block = side * side
        assert coordinates.size % block == 0, "coordinates must be K*(2*box1+1)^2"
        num_tweezers = coordinates.size // block

        error_signal = 1.0
        ok_in_a_row = 0

        for ii in range(1, iterations + 1):
            A_mod = A_in.astype(np.complex64, copy=False) * np.exp(1j * psi)
            A_out = fftshift(fft2(ifftshift(A_mod)))
            psi_out = np.mod(np.angle(A_out), 2 * np.pi)
            B_out = np.abs(A_out).astype(np.float64, copy=False)

            B_out_flat = B_out.ravel()

            if coordinates.size:
                B_out0 = B_out_flat[coordinates]
                B_mean = float(np.mean(B_out0))

                B_out0_reshaped = B_out0.reshape(num_tweezers, block).T
                B_out_box = np.mean(B_out0_reshaped, axis=0)

                weight = B_mean / (B_out_box + 1e-12)

                error_signal = np.std(weight)
                print(f'Iteration {ii}: error signal = {error_signal:.5f}')

                if error_signal < tol:
                    ok_in_a_row += 1
                else:
                    ok_in_a_row = 0
                if ok_in_a_row >= 3:
                    print(f"[GS] Early stop at iter {ii}: error={error_signal:.6g} < {tol} for {ok_in_a_row} iters")
                    break

                weight_expanded = np.repeat(weight, block)
                g_flat = g.ravel()
                g_flat[coordinates] = weight_expanded * g_flat[coordinates]
                g = g_flat.reshape(g.shape)

            A_iter = g * B_target * np.exp(1j * psi_out)
            A_new = fftshift(ifft2(ifftshift(A_iter)))
            psi = np.mod(np.angle(A_new), 2 * np.pi).astype(np.float32)
            psi *= pad

        self.convergence = 1 if error_signal <= 0.01 else 0
        return psi

    # === Simple multi-plane GS (regular GS, split into z-planes) =================

    def assign_planes_from_tilt(self, *, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0,
                                n_planes: int = 5) -> None:
        assert self.target_xy_um is not None, "Call set_target_grid() first."
        assert self._k_over_2f2 is not None, "Call set_optics() first."

        # 1. compute each tweezer's z offset in the atom plane
        tx = np.tan(np.deg2rad(float(tilt_x_deg)))
        ty = np.tan(np.deg2rad(float(tilt_y_deg)))

        xy = self.target_xy_um.astype(np.float64)
        x0, y0 = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
        # z_i is *image-plane defocus* (µm) implied by the tilt at 2f
        z_per_spot = tx * (xy[:, 0] - x0) + ty * (xy[:, 1] - y0)
        self._z_per_spot = z_per_spot.astype(np.float32)

        z_min_um = float(np.min(z_per_spot))
        z_max_um = float(np.max(z_per_spot))

        # 2. create/discretize n_planes at different z values & assign
        self._z_planes = np.linspace(float(z_min_um), float(z_max_um), int(n_planes)).astype(np.float32)
        # assign each tweezer/z_spot to an z_plane (via an index of the above array)
        idx = np.abs(self._z_planes[None, :] - z_per_spot[:, None]).argmin(axis=1).astype(np.int32)
        self._members = [np.where(idx == p)[0].tolist() for p in range(n_planes)]

        # DIAGNOSTIC
        print(f"  [assign_planes_from_tilt] tx={tx:.6f}, ty={ty:.6f}")
        print(f"  [assign_planes_from_tilt] z_per_spot range: [{z_min_um:.2f}, {z_max_um:.2f}] µm")
        print(f"  [assign_planes_from_tilt] _z_planes: {self._z_planes}")
        for p in range(n_planes):
            print(f"    Plane {p}: z={self._z_planes[p]:+.2f} µm, {len(self._members[p])} tweezers")

        # 3. pre-compute a quadratic defocus per plane
        # phi_p(X,Y) = (k / (2 f^2)) * z_p * (X^2 + Y^2)
        assert (self._pupil_X2 is not None) and (self._pupil_Y2 is not None)
        self._phi_planes = []
        quad = (self._pupil_X2 + self._pupil_Y2).astype(np.float32, copy=False)
        f = float(self.focal_length_um)
        k = self._k
        
        for p_idx, z_p in enumerate(self._z_planes):
            # PARAXIAL defocus formula: φ = +(k/(2f²)) · z · R²
            # SIGN: Positive z (focus farther) needs POSITIVE (diverging) phase
            # Physical interpretation: A converging lens has φ=-kR²/(2f). To push
            # the focus farther (z>0), we add positive phase to make it less converging.
            # This is consistent with Goodman "Fourier Optics" Ch. 5.
            # The "exact" formula φ = -k·z·R²/(2f(f-z)) incorrectly scales
            # corrections by z/f, making them vanish for large f.
            z_val = float(z_p)
            phase = +(k / (2.0 * f * f)) * z_val * quad

            # a phi correction for each z_plane
            # pupil_X2 are in the SLM plane
            self._phi_planes.append(phase.astype(np.float32, copy=False))

            # DIAGNOSTIC
            if p_idx == 0 or p_idx == len(self._z_planes) - 1:
                phase_rms = np.sqrt(np.mean(phase**2))
                phase_max = np.max(np.abs(phase))
                print(f"    Plane {p_idx} (z={z_val:+.2f} µm): phase RMS={phase_rms:.4f} rad, max={phase_max:.4f} rad")

    def run_gs_multiplane_v3(self, *, iterations: int = 50, Gg: float = 0.6, verbose: bool = True, tol: float = 1e-4) -> None:
        """
        Multi-plane GS with per-plane targets and configurable early-stop tolerance.
        Stops when std(weight) < tol for 3 consecutive iterations.
        """
        assert self.A_in is not None and self.pad is not None
        assert self.A_target is not None and self.psi0 is not None
        assert self.coordinates is not None and self.height_corr2 is not None
        assert hasattr(self, "_phi_planes") and hasattr(self, "_members")

        side = 2 * self.box1 + 1
        block = side * side
        num_tweezers = self.coordinates.size // block
        P = len(self._phi_planes)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Multi-plane GS v3: {num_tweezers} tweezers, {P} planes")
            for p in range(P):
                print(f"  Plane {p} @ z={self._z_planes[p]:+.1f} µm: {len(self._members[p])} tweezers")
            print(f"{'=' * 70}")

        g = np.ones_like(self.A_target, dtype=self.A_target.dtype)
        B_target_global = np.abs(self.A_target)
        B_global_flat = B_target_global.ravel()

        B_target_planes = []
        for p in range(P):
            B_target_p = np.zeros_like(B_target_global, dtype=np.float32)
            if len(self._members[p]) > 0:
                B_target_flat = B_target_p.ravel()
                for s in self._members[p]:
                    idx0 = s * block
                    idxs = self.coordinates[idx0:idx0 + block]
                    B_target_flat[idxs] = B_global_flat[idxs]
                B_target_p = B_target_flat.reshape(B_target_p.shape)
            B_target_planes.append(B_target_p)

        psi = np.zeros_like(self.A_target, dtype=np.float32)
        my = (psi.shape[0] - self.psi0.shape[0]) // 2
        mx = (psi.shape[1] - self.psi0.shape[1]) // 2
        psi[my:my + self.psi0.shape[0], mx:mx + self.psi0.shape[1]] = self.psi0

        error_signal = 1.0
        ok_in_a_row = 0

        for ii in range(1, iterations + 1):
            A_mod = self.A_in.astype(np.complex64, copy=False) * np.exp(1j * psi)

            B_out_all = np.zeros_like(self.A_target, dtype=np.float64)
            exp_i_psi_per_plane = [None] * P

            for p in range(P):
                if not self._members[p]:
                    continue
                phi_p = self._phi_planes[p]
                if abs(float(self._z_planes[p])) < 1e-6:
                    A_out_p = fftshift(fft2(ifftshift(A_mod)))
                else:
                    A_out_p = fftshift(fft2(ifftshift(A_mod * np.exp(1j * phi_p))))

                exp_i_psi_per_plane[p] = np.exp(1j * np.angle(A_out_p)).astype(np.complex64)

                B_out_p = np.abs(A_out_p).astype(np.float64, copy=False)
                B_out_flat = B_out_p.ravel()
                B_all_flat = B_out_all.ravel()
                for s in self._members[p]:
                    idx0 = s * block
                    idxs = self.coordinates[idx0:idx0 + block]
                    B_all_flat[idxs] = B_out_flat[idxs]
            B_out_all = B_all_flat.reshape(B_out_all.shape)

            g = np.ones_like(self.A_target, dtype=self.A_target.dtype) if 'g' not in locals() else g
            B_out_flat = B_out_all.ravel()

            if ii > 1 and self.coordinates.size:
                correction = (1.0 - Gg) + Gg * np.sqrt(self.height_corr2)
                correction = np.clip(correction, 1e-3, 1e3)
                B_out_flat[self.coordinates] = B_out_flat[self.coordinates] / correction.ravel()

            if self.coordinates.size:
                B_out0 = B_out_flat[self.coordinates]
                B_mean = float(np.mean(B_out0[B_out0 > 0]))
                B_out0_reshaped = B_out0.reshape(num_tweezers, block).T
                B_out_box = np.mean(B_out0_reshaped, axis=0)
                weight = B_mean / (B_out_box + 1e-12)
                weight_expanded = np.repeat(weight, block)
                g_flat = g.ravel()
                g_flat[self.coordinates] = weight_expanded * g_flat[self.coordinates]
                g = g_flat.reshape(g.shape)
                error_signal = float(np.std(weight))
                print(f'Iteration {ii}: error signal = {error_signal:.5f}')

                if error_signal < tol:
                    ok_in_a_row += 1
                else:
                    ok_in_a_row = 0
                if ok_in_a_row >= 3:
                    print(f"[GS] Early stop at iter {ii}: error={error_signal:.6g} < {tol} for {ok_in_a_row} iters")
                    break

            A_new_accum = np.zeros_like(A_mod, dtype=np.complex128)
            for p in range(P):
                if not self._members[p]:
                    continue
                exp_i_psi = exp_i_psi_per_plane[p]
                A_iter_p = g * B_target_planes[p] * exp_i_psi
                A_back_p = fftshift(ifft2(ifftshift(A_iter_p)))
                if abs(float(self._z_planes[p])) >= 1e-6:
                    A_back_p = A_back_p * np.exp(-1j * self._phi_planes[p])
                A_new_accum += A_back_p

            psi = np.mod(np.angle(A_new_accum), 2 * np.pi).astype(np.float32)
            psi *= self.pad

        scal = int(self.config.scal)
        ypix, xpix = self.A_in.shape[0] // scal, self.A_in.shape[1] // scal
        mid_y_start = int((scal - 1) * ypix / 2)
        mid_y_end = int((scal + 1) * ypix / 2)
        mid_x_start = int((scal - 1) * xpix / 2)
        mid_x_end = int((scal + 1) * xpix / 2)
        phase_mask = psi[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
        self.phase_mask = np.mod(phase_mask, 2 * np.pi)

        self.convergence = 1 if error_signal <= 0.01 else 0

        if verbose:
            print(f"{'=' * 70}")
            print(f"✓ Convergence: {self.convergence}, Final error: {error_signal:.5f}")
            print(f"{'=' * 70}")

        _cleanup_memory(verbose=False)

    def run_gs_multiplane_adaptive(self, *, iterations: int = 50, Gg: float = 0.6,
                                    z_scan_every: int = 5, z_scan_range_um: float = 100.0,
                                    z_scan_steps: int = 21, peak_sharpness_threshold: float = 2.0,
                                    z_correction_factor: float = 0.3,
                                    verbose: bool = True, tol: float = 1e-4) -> None:
        """
        Adaptive multi-plane GS that periodically measures where each tweezer actually focuses
        and adjusts defocus phases to improve z-accuracy.

        Args:
            iterations: Maximum number of GS iterations
            Gg: GS feedback gain
            z_scan_every: Perform z-scan every N iterations (e.g., 5)
            z_scan_range_um: Z-range to scan around target (±this value)
            z_scan_steps: Number of z-slices to scan
            peak_sharpness_threshold: Minimum peak/mean ratio to consider a tweezer "good"
            z_correction_factor: How aggressively to correct z-errors (0-1, smaller = more gradual)
            verbose: Print progress
            tol: Early-stop tolerance
        """
        assert self.A_in is not None and self.pad is not None
        assert self.A_target is not None and self.psi0 is not None
        assert self.coordinates is not None and self.height_corr2 is not None
        assert hasattr(self, "_phi_planes") and hasattr(self, "_members")
        assert hasattr(self, "_z_per_spot") and hasattr(self, "target_xy_um")
        assert hasattr(self, "wavelength_um") and hasattr(self, "focal_length_um")

        side = 2 * self.box1 + 1
        block = side * side
        num_tweezers = self.coordinates.size // block
        P = len(self._phi_planes)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Adaptive Multi-plane GS: {num_tweezers} tweezers, {P} planes")
            print(f"  Z-scan every {z_scan_every} iterations, range ±{z_scan_range_um:.1f} µm")
            print(f"  Peak sharpness threshold: {peak_sharpness_threshold:.2f}")
            for p in range(P):
                print(f"  Plane {p} @ z={self._z_planes[p]:+.1f} µm: {len(self._members[p])} tweezers")
            print(f"{'=' * 70}")

        # Initialize per-tweezer quality weights (1.0 = full weight)
        tweezer_quality = np.ones(num_tweezers, dtype=np.float32)

        # Precompute focal plane pixel coordinates
        H, W = self.A_in.shape
        pixel_um = self.params.pixel_um * int(self.config.scal)
        lam = float(self.wavelength_um)
        f = float(self.focal_length_um)
        k = 2.0 * np.pi / lam
        px_focal_um = (lam * f) / (W * pixel_um)

        # Pupil coordinates for defocus
        yy = (np.arange(H) - H / 2) * pixel_um
        xx = (np.arange(W) - W / 2) * pixel_um
        X_pupil, Y_pupil = np.meshgrid(xx, yy)
        R2_pupil = X_pupil**2 + Y_pupil**2

        g = np.ones_like(self.A_target, dtype=self.A_target.dtype)
        B_target_global = np.abs(self.A_target)
        B_global_flat = B_target_global.ravel()

        B_target_planes = []
        for p in range(P):
            B_target_p = np.zeros_like(B_target_global, dtype=np.float32)
            if len(self._members[p]) > 0:
                B_target_flat = B_target_p.ravel()
                for s in self._members[p]:
                    idx0 = s * block
                    idxs = self.coordinates[idx0:idx0 + block]
                    B_target_flat[idxs] = B_global_flat[idxs]
                B_target_p = B_target_flat.reshape(B_target_p.shape)
            B_target_planes.append(B_target_p)

        psi = np.zeros_like(self.A_target, dtype=np.float32)
        my = (psi.shape[0] - self.psi0.shape[0]) // 2
        mx = (psi.shape[1] - self.psi0.shape[1]) // 2
        psi[my:my + self.psi0.shape[0], mx:mx + self.psi0.shape[1]] = self.psi0

        error_signal = 1.0
        ok_in_a_row = 0

        for ii in range(1, iterations + 1):
            A_mod = self.A_in.astype(np.complex64, copy=False) * np.exp(1j * psi)

            # ========== ADAPTIVE Z-SCANNING ==========
            if ii > 1 and ii % z_scan_every == 0:
                if verbose:
                    print(f"\n  [Adaptive] Iteration {ii}: Scanning focal positions...")

                # Build z-scan array (global, same for all tweezers)
                z_min_global = float(np.min(self._z_per_spot)) - z_scan_range_um
                z_max_global = float(np.max(self._z_per_spot)) + z_scan_range_um
                z_scan = np.linspace(z_min_global, z_max_global, z_scan_steps)

                # Precompute focal plane coordinates (once)
                x_focal_um = (np.arange(W) - W/2) * px_focal_um
                y_focal_um = (np.arange(H) - H/2) * px_focal_um

                # Build full 3D intensity volume: I(z, y, x)
                # This is the key efficiency gain: only z_scan_steps FFTs total!
                I_volume = np.zeros((z_scan_steps, H, W), dtype=np.float32)

                if verbose:
                    print(f"    Building X-Z maps ({z_scan_steps} z-slices)...", end='', flush=True)

                for zi, z_test in enumerate(z_scan):
                    phase_defocus = +(k / (2.0 * f * f)) * z_test * R2_pupil
                    A_out = fftshift(fft2(ifftshift(A_mod * np.exp(1j * phase_defocus))))
                    I_volume[zi, :, :] = np.abs(A_out)**2

                if verbose:
                    print(" Done!")

                # Now extract intensity curves for each tweezer
                z_adjustments = []
                quality_scores = []

                for tw_idx in range(num_tweezers):
                    target_z = float(self._z_per_spot[tw_idx])
                    target_xy = self.target_xy_um[tw_idx]  # (x, y) in microns

                    # Find nearest pixel to target position
                    ix = np.argmin(np.abs(x_focal_um - target_xy[0]))
                    iy = np.argmin(np.abs(y_focal_um - target_xy[1]))

                    # Extract intensity in small box around target (3x3 box, take max)
                    box_size = 3
                    iy_min = max(0, iy - box_size//2)
                    iy_max = min(H, iy + box_size//2 + 1)
                    ix_min = max(0, ix - box_size//2)
                    ix_max = min(W, ix + box_size//2 + 1)

                    # Extract z-curve at this (x,y) position
                    # I_volume shape: (z, y, x)
                    # Take max over small spatial box for each z
                    intensities = np.max(I_volume[:, iy_min:iy_max, ix_min:ix_max], axis=(1, 2))

                    # Analyze peak
                    peak_idx = np.argmax(intensities)
                    peak_z = z_scan[peak_idx]
                    peak_val = intensities[peak_idx]
                    mean_val = np.mean(intensities[intensities > 0])

                    # Peak sharpness = ratio of peak to mean
                    peak_sharpness = peak_val / (mean_val + 1e-12)

                    # Z-error
                    z_error = peak_z - target_z
                    z_adjustments.append(z_error)
                    quality_scores.append(peak_sharpness)

                    # Update quality weight
                    if peak_sharpness < peak_sharpness_threshold:
                        tweezer_quality[tw_idx] *= 0.8  # Reduce weight for low-quality tweezers
                    else:
                        tweezer_quality[tw_idx] = min(1.0, tweezer_quality[tw_idx] * 1.1)  # Slowly restore

                # Apply z-corrections
                z_adjustments = np.array(z_adjustments)
                quality_scores = np.array(quality_scores)

                # Gradual correction
                self._z_per_spot += z_correction_factor * z_adjustments

                # Recompute defocus phases for each plane
                for p in range(P):
                    if len(self._members[p]) > 0:
                        z_p = float(self._z_planes[p])
                        self._phi_planes[p] = (k / (2.0 * f * f)) * z_p * R2_pupil

                if verbose:
                    n_good = np.sum(quality_scores >= peak_sharpness_threshold)
                    print(f"    Z-errors: RMS={np.std(z_adjustments):.2f} µm, max={np.max(np.abs(z_adjustments)):.2f} µm")
                    print(f"    Quality: {n_good}/{num_tweezers} tweezers above threshold")
                    print(f"    Mean sharpness: {np.mean(quality_scores):.2f}")

            # ========== STANDARD GS ITERATION ==========
            B_out_all = np.zeros_like(self.A_target, dtype=np.float64)
            exp_i_psi_per_plane = [None] * P

            for p in range(P):
                if not self._members[p]:
                    continue
                phi_p = self._phi_planes[p]
                if abs(float(self._z_planes[p])) < 1e-6:
                    A_out_p = fftshift(fft2(ifftshift(A_mod)))
                else:
                    A_out_p = fftshift(fft2(ifftshift(A_mod * np.exp(1j * phi_p))))

                exp_i_psi_per_plane[p] = np.exp(1j * np.angle(A_out_p)).astype(np.complex64)

                B_out_p = np.abs(A_out_p).astype(np.float64, copy=False)
                B_out_flat = B_out_p.ravel()
                B_all_flat = B_out_all.ravel()
                for s in self._members[p]:
                    idx0 = s * block
                    idxs = self.coordinates[idx0:idx0 + block]
                    B_all_flat[idxs] = B_out_flat[idxs]
            B_out_all = B_all_flat.reshape(B_out_all.shape)

            g = np.ones_like(self.A_target, dtype=self.A_target.dtype) if 'g' not in locals() else g
            B_out_flat = B_out_all.ravel()

            if ii > 1 and self.coordinates.size:
                correction = (1.0 - Gg) + Gg * np.sqrt(self.height_corr2)
                correction = np.clip(correction, 1e-3, 1e3)
                B_out_flat[self.coordinates] = B_out_flat[self.coordinates] / correction.ravel()

            if self.coordinates.size:
                B_out0 = B_out_flat[self.coordinates]
                B_mean = float(np.mean(B_out0[B_out0 > 0]))
                B_out0_reshaped = B_out0.reshape(num_tweezers, block).T
                B_out_box = np.mean(B_out0_reshaped, axis=0)

                # Apply quality weights to the weight calculation
                weight = B_mean / (B_out_box + 1e-12)
                # NOTE: Quality-based weight modulation disabled - causes non-uniform intensity
                # when all tweezers have poor sharpness due to large Rayleigh range
                # weight = weight * tweezer_quality  # DISABLED

                weight_expanded = np.repeat(weight, block)
                g_flat = g.ravel()
                g_flat[self.coordinates] = weight_expanded * g_flat[self.coordinates]
                g = g_flat.reshape(g.shape)
                error_signal = float(np.std(weight))

                if ii % z_scan_every != 0 or ii == 1:  # Don't print during z-scan iterations
                    print(f'  Iteration {ii}: error signal = {error_signal:.5f}')

                if error_signal < tol:
                    ok_in_a_row += 1
                else:
                    ok_in_a_row = 0
                if ok_in_a_row >= 3:
                    print(f"  [GS] Early stop at iter {ii}: error={error_signal:.6g} < {tol} for {ok_in_a_row} iters")
                    break

            A_new_accum = np.zeros_like(A_mod, dtype=np.complex128)
            for p in range(P):
                if not self._members[p]:
                    continue
                exp_i_psi = exp_i_psi_per_plane[p]
                A_iter_p = g * B_target_planes[p] * exp_i_psi
                A_back_p = fftshift(ifft2(ifftshift(A_iter_p)))
                if abs(float(self._z_planes[p])) >= 1e-6:
                    A_back_p = A_back_p * np.exp(-1j * self._phi_planes[p])
                A_new_accum += A_back_p

            psi = np.mod(np.angle(A_new_accum), 2 * np.pi).astype(np.float32)
            psi *= self.pad

        scal = int(self.config.scal)
        ypix, xpix = self.A_in.shape[0] // scal, self.A_in.shape[1] // scal
        mid_y_start = int((scal - 1) * ypix / 2)
        mid_y_end = int((scal + 1) * ypix / 2)
        mid_x_start = int((scal - 1) * xpix / 2)
        mid_x_end = int((scal + 1) * xpix / 2)
        phase_mask = psi[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
        self.phase_mask = np.mod(phase_mask, 2 * np.pi)

        self.convergence = 1 if error_signal <= 0.01 else 0

        if verbose:
            print(f"{'=' * 70}")
            print(f"✓ Convergence: {self.convergence}, Final error: {error_signal:.5f}")
            print(f"  Tweezer quality: min={np.min(tweezer_quality):.2f}, mean={np.mean(tweezer_quality):.2f}")
            print(f"{'=' * 70}")

        _cleanup_memory(verbose=False)

    def save_pickle(self, *, out_dir: str | Path = "/Users/nadinemeister/PyCharmMiscProject/python_SLM_3d/out",
                    label: Optional[str] = None) -> ResultBundle:
        assert self.phase_mask is not None and self.tweezlist is not None and self.tweezer_mask is not None
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = (label or "")
        fname = f"{stem}_{now}.pkl" if stem else f"tweezers_{now}.pkl"
        out_path = str(Path(out_dir) / fname)
        bundle = ResultBundle(
            phase_mask=self.phase_mask,
            A_in=self.A_in,
            tweezlist=self.tweezlist,
            weights=self.height_corr if self.height_corr is not None else np.empty((0, 1)),
            tweezer_mask=self.tweezer_mask,
            convergence=int(1),
            file=out_path,
        )
        with open(out_path, "wb") as f:
            f.write(pickle.dumps(bundle))
        self.last_file = out_path
        return bundle

    # ALSO add this method to the SLMTweezers class:
    def clear_large_arrays(self):
        """
        Manually clear large intermediate arrays.
        Call this after save_pickle() to free memory.
        """
        # Clear intermediate computation arrays
        self.A_in = None
        self.A_target = None
        self.pad = None
        self.psi0 = None
        self.filter_2d = None

        # Clear pupil coordinates (large arrays)
        self._pupil_X2 = None
        self._pupil_Y2 = None
        self._pupil_XY = None

        # Keep only the final phase mask (this is small)
        # self.phase_mask is kept

        # Force cleanup
        _cleanup_memory_aggressive(verbose=False)


# Add this to the BOTTOM of your slm_tweezers_class_WITH_AUTO_CLEANUP.py file
# Or replace the _cleanup_memory function with this version

def _cleanup_memory_aggressive(verbose: bool = False):
    """
    AGGRESSIVE memory cleanup - actually works!
    """
    if verbose:
        print("  🧹 Aggressive cleanup...")

    # 1. Close ALL matplotlib figures (they hold array references!)
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        if verbose:
            print("    ✓ Closed matplotlib figures")
    except:
        pass

    # 2. Force Python garbage collection (3 times for good measure)
    import gc
    for _ in range(3):
        gc.collect()

    # 3. Clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        if verbose:
            print("    ✓ Cleared PyTorch cache")
    except:
        pass

    # 4. Clear CuPy memory pool
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        if verbose:
            print("    ✓ Cleared CuPy cache")
    except:
        pass

    # 5. Final GC sweep
    gc.collect()

    if verbose:
        print("  ✓ Cleanup complete")


