"""
fwi_utils_documented.py

Utility functions for seismic modeling, full-waveform inversion (FWI), diffusion
image prior workflows, patch-based processing, and common image-quality metrics.

The original script mixed utilities, plotting code, and global variables. This
version keeps the same functionality but improves readability, documentation,
input validation, and naming consistency where possible.

Notes
-----
1. Deepwave is required only for functions/classes that run wave-equation
   propagation: ``create_data`` and ``PhySimulator``.
2. Most tensors are assumed to use the Deepwave convention:
   receiver/source amplitudes: [nt, n_shots, n_receivers or n_sources].
3. Velocity models are assumed to be 2D tensors/arrays with shape [nz, nx], where
   the first axis is depth and the second axis is horizontal position.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io as spio
import scipy.ndimage
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.signal import iirfilter, sosfilt, zpk2sos
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
try:
    import deepwave
except ImportError:  # pragma: no cover - only needed when Deepwave is unavailable.
    deepwave = None

ArrayLike = Union[np.ndarray, Tensor]


# =============================================================================
# Wavelets, interpolation, and filtering
# =============================================================================

def interp_depth_to_256(
    model: Tensor,
    nz_target: int = 256,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    """Interpolate a 2D model along the depth axis.

    Parameters
    ----------
    model:
        Input 2D tensor with shape ``[nz, nx]``.
    nz_target:
        Target number of depth samples. The horizontal dimension is preserved.
    mode:
        Interpolation mode passed to ``torch.nn.functional.interpolate``.
    align_corners:
        Value passed to ``F.interpolate`` for linear/bilinear modes.

    Returns
    -------
    Tensor
        Interpolated model with shape ``[nz_target, nx]``.
    """
    if model.ndim != 2:
        raise ValueError(f"model must be 2D [nz, nx], but got shape {tuple(model.shape)}")

    nz, nx = model.shape
    model_4d = model[None, None, :, :]  # [1, 1, nz, nx]
    output = F.interpolate(
        model_4d,
        size=(nz_target, nx),
        mode=mode,
        align_corners=align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
    )
    return output[0, 0]


def ricker_wavelet(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a Ricker wavelet.

    Parameters
    ----------
    freq:
        Peak/central frequency in Hz.
    length:
        Number of time samples.
    dt:
        Sampling interval in seconds.
    peak_time:
        Time of the wavelet peak in seconds.
    dtype:
        Output tensor dtype.
    device:
        Optional PyTorch device.

    Returns
    -------
    Tensor
        Ricker wavelet with shape ``[length]``.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if freq <= 0:
        raise ValueError("freq must be positive.")

    t = torch.arange(length, dtype=dtype, device=device) * dt - peak_time
    arg = (math.pi * freq * t) ** 2
    return (1.0 - 2.0 * arg) * torch.exp(-arg)


# Backward-compatible alias used in older notebooks.
rickerwav = ricker_wavelet


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert a tensor/array to a NumPy array without modifying the input."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def highpass_filter(freq, wavelet, dt, pad=None):
    """
    Filter out low frequency

    Parameters
    ----------
    freq : :obj:`int`
    Cut-off frequency
    wavelet : :obj:`torch.Tensor`
    Tensor of wavelet
    dt : :obj:`float32`
    Time sampling
    Returns
    -------
    : :obj:`torch.Tensor`
    Tensor of highpass frequency wavelet
    """
    
    if pad is not None:
        wavelet = F.pad(wavelet, pad)
    
    sos = signal.butter(8,  freq / (0.5 * (1 / dt)), 'hp', output='sos')
    wavelet = torch.tensor( signal.sosfiltfilt(sos, wavelet,axis=-1).copy(),dtype=torch.float32)
    if pad is not None:
        if pad[1]:
            wavelet = wavelet[..., pad[0]:-pad[1]]
        else:
            wavelet = wavelet[..., pad[0]:]
        
    return wavelet


def seismic_filter(
    data: np.ndarray,
    filter_type: str,
    freqmin: Optional[float] = None,
    freqmax: Optional[float] = None,
    df: Optional[float] = None,
    corners: int = 4,
    zerophase: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Apply a Butterworth band-pass, low-pass, or high-pass filter.

    Parameters
    ----------
    data:
        Input array to filter.
    filter_type:
        One of ``"bandpass"``, ``"lowpass"``, or ``"highpass"``.
    freqmin:
        Low corner frequency in Hz. Required for ``bandpass`` and ``highpass``.
    freqmax:
        High corner frequency in Hz. Required for ``bandpass`` and ``lowpass``.
    df:
        Sampling rate in Hz.
    corners:
        Butterworth filter order.
    zerophase:
        If True, filter forward and backward to approximately remove phase shift.
    axis:
        Axis along which to filter.

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    filter_type = filter_type.lower()
    if filter_type not in {"bandpass", "lowpass", "highpass"}:
        raise ValueError("filter_type must be 'bandpass', 'lowpass', or 'highpass'.")
    if df is None or df <= 0:
        raise ValueError("df must be a positive sampling rate in Hz.")

    if filter_type == "bandpass":
        if freqmin is None or freqmax is None:
            raise ValueError("freqmin and freqmax are required for bandpass filtering.")
        return bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
    if filter_type == "lowpass":
        if freqmax is None:
            raise ValueError("freqmax is required for lowpass filtering.")
        return lowpass(data, freqmax, df, corners, zerophase, axis)
    if freqmin is None:
        raise ValueError("freqmin is required for highpass filtering.")
    return highpass(data, freqmin, df, corners, zerophase, axis)


def bandpass(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    df: float,
    corners: int = 4,
    zerophase: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter."""
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe

    if high >= 1.0:
        warnings.warn(
            f"Selected high corner frequency ({freqmax}) is at or above Nyquist ({fe}). "
            "Applying a high-pass filter instead.",
            RuntimeWarning,
        )
        return highpass(data, freqmin, df, corners, zerophase, axis)
    if low <= 0 or low >= 1:
        raise ValueError("freqmin must be between 0 and the Nyquist frequency.")
    if low >= high:
        raise ValueError("freqmin must be smaller than freqmax.")

    z, p, k = iirfilter(corners, [low, high], btype="band", ftype="butter", output="zpk")
    sos = zpk2sos(z, p, k)
    if zerophase:
        first = sosfilt(sos, data, axis=axis)
        return sosfilt(sos, np.flip(first, axis=axis), axis=axis).take(indices=range(first.shape[axis]-1, -1, -1), axis=axis)
    return sosfilt(sos, data, axis=axis)


def lowpass(
    data: np.ndarray,
    freq: float,
    df: float,
    corners: int = 4,
    zerophase: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    fe = 0.5 * df
    f = freq / fe
    if f >= 1.0:
        warnings.warn("Selected corner frequency is above Nyquist. Using Nyquist as corner.", RuntimeWarning)
        f = 0.999
    if f <= 0:
        raise ValueError("freq must be positive.")

    z, p, k = iirfilter(corners, f, btype="lowpass", ftype="butter", output="zpk")
    sos = zpk2sos(z, p, k)
    if zerophase:
        first = sosfilt(sos, data, axis=axis)
        return sosfilt(sos, np.flip(first, axis=axis), axis=axis).take(indices=range(first.shape[axis]-1, -1, -1), axis=axis)
    return sosfilt(sos, data, axis=axis)


def highpass(
    data: np.ndarray,
    freq: float,
    df: float,
    corners: int = 4,
    zerophase: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Apply a Butterworth high-pass filter."""
    fe = 0.5 * df
    f = freq / fe
    if f <= 0 or f >= 1.0:
        raise ValueError("freq must be between 0 and the Nyquist frequency.")

    z, p, k = iirfilter(corners, f, btype="highpass", ftype="butter", output="zpk")
    sos = zpk2sos(z, p, k)
    if zerophase:
        first = sosfilt(sos, data, axis=axis)
        return sosfilt(sos, np.flip(first, axis=axis), axis=axis).take(indices=range(first.shape[axis]-1, -1, -1), axis=axis)
    return sosfilt(sos, data, axis=axis)


# =============================================================================
# Geometry, source amplitudes, and Deepwave modeling
# =============================================================================

def create_sr(
    num_shots: int,
    num_sources_per_shot: int,
    num_receivers_per_shot: int,
    num_dims: int,
    source_spacing: float,
    receiver_spacing: float,
    source_depth: float,
    receiver_depth: float,
) -> Tuple[Tensor, Tensor]:
    """Create Deepwave-style source and receiver coordinate tensors.

    Returns
    -------
    x_s:
        Source coordinates with shape ``[num_shots, num_sources_per_shot, num_dims]``.
    x_r:
        Receiver coordinates with shape ``[num_shots, num_receivers_per_shot, num_dims]``.
    """
    if num_dims != 2:
        raise ValueError("This utility currently assumes 2D coordinates [z, x].")

    x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims, dtype=torch.float32)
    x_s[:, 0, 0] = float(source_depth)
    x_s[:, 0, 1] = torch.arange(1, num_shots + 1, dtype=torch.float32) * source_spacing

    x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims, dtype=torch.float32)
    x_r[:, :, 0] = float(receiver_depth)
    receiver_x = torch.arange(1, num_receivers_per_shot + 1, dtype=torch.float32) * receiver_spacing
    x_r[:, :, 1] = receiver_x.repeat(num_shots, 1)
    return x_s, x_r


# Backward-compatible alias.
createSR = create_sr


def create_source_amplitudes(
    peak_freq: float,
    nt: int,
    dt: float,
    peak_source_time: float,
    num_shots: int,
    num_sources_per_shot: int,
    as_tensor: bool = False,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, Tensor]:
    """Create repeated Ricker source amplitudes.

    Returns shape ``[nt, num_shots, num_sources_per_shot]``.
    """
    wavelet = ricker_wavelet(peak_freq, nt, dt, peak_source_time, device=device)
    source = wavelet.reshape(-1, 1, 1).repeat(1, num_shots, num_sources_per_shot)
    return source if as_tensor else source.detach().cpu().numpy()


createSourceAmp = create_source_amplitudes


def create_filter_source_amplitudes(
    peak_freq: float,
    nt: int,
    dt: float,
    peak_source_time: float,
    num_shots: int,
    num_sources_per_shot: int,
    use_filter: bool,
    filter_type: str,
    freqmin: Optional[float],
    freqmax: Optional[float],
    corners: int,
    df: float,
) -> np.ndarray:
    """Create Ricker source amplitudes and optionally filter them.

    Returns
    -------
    np.ndarray
        Source amplitudes with shape ``[nt, num_shots, num_sources_per_shot]``.
    """
    wavelet = ricker_wavelet(peak_freq, nt, dt, peak_source_time).cpu().numpy()
    if use_filter:
        wavelet = seismic_filter(wavelet, filter_type, freqmin, freqmax, df, corners)
    return np.tile(wavelet.reshape(-1, 1, 1), [1, num_shots, num_sources_per_shot])


createFilterSourceAmp = create_filter_source_amplitudes


def create_initial_model(
    model_true: Tensor,
    mode: str,
    lipar: float,
    fix_value_depth: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Create an initial velocity model for inversion.

    Parameters
    ----------
    model_true:
        True velocity model with shape ``[nz, nx]``.
    mode:
        Initialization strategy: ``"line"``, ``"lineminmax"``, ``"constant"``/``"const"``, or ``"GS"``.
    lipar:
        Parameter controlling either the linear scaling or Gaussian smoothing sigma.
    fix_value_depth:
        Number of shallow depth samples copied from the true model and kept fixed.
    device:
        Device for the trainable model.

    Returns
    -------
    model:
        Trainable initial model on ``device`` with ``requires_grad=True``.
    model_init:
        Initial model tensor on CPU.
    """
    mode = mode.lower()
    if mode not in {"line", "lineminmax", "constant", "const", "gs"}:
        raise ValueError("mode must be 'line', 'lineminmax', 'constant'/'const', or 'GS'.")

    true_np = _to_numpy(model_true).astype(np.float32)
    nz, nx = true_np.shape
    if fix_value_depth < 0 or fix_value_depth >= nz:
        raise ValueError("fix_value_depth must be in [0, nz-1].")

    fixed = true_np[:fix_value_depth, :] if fix_value_depth > 0 else None
    active_nz = nz - fix_value_depth

    if mode == "line":
        start = true_np[fix_value_depth, nx // 2]
        end = true_np[-1, nx // 2] * lipar
        value = np.linspace(start, end, num=active_nz, dtype=np.float32).reshape(-1, 1).repeat(nx, axis=1)
    elif mode == "lineminmax":
        value = np.linspace(true_np.min() * lipar, true_np.max(), num=active_nz, dtype=np.float32).reshape(-1, 1).repeat(nx, axis=1)
    elif mode in {"constant", "const"}:
        value = np.full((active_nz, nx), true_np[fix_value_depth, nx // 2], dtype=np.float32)
    else:  # Gaussian smoothing
        value = scipy.ndimage.gaussian_filter(true_np[fix_value_depth:, :], sigma=lipar).astype(np.float32)

    model_init_np = np.concatenate([fixed, value], axis=0) if fixed is not None else value
    model_init = torch.as_tensor(model_init_np, dtype=torch.float32)
    model = model_init.clone().to(device).requires_grad_(True)
    return model, model_init


createInitialModel = create_initial_model


def _require_deepwave() -> None:
    """Raise a clear error if Deepwave is not installed."""
    if deepwave is None:
        raise ImportError("Deepwave is required for this function. Install/import deepwave first.")


def create_data(
    model: Tensor,
    dx: Union[float, Sequence[float]],
    source_amplitudes: Tensor,
    x_s: Tensor,
    x_r: Tensor,
    dt: float,
    pml_width,
    order: int,
    survey_pad,
    device: torch.device,
) -> Tensor:
    """Model seismic data using Deepwave's scalar propagator.

    Returns receiver amplitudes with shape ``[nt, num_shots, num_receivers]`` on CPU.
    """
    _require_deepwave()
    prop = deepwave.scalar.Propagator({"vp": model.to(device)}, dx, pml_width, order, survey_pad)
    receiver_amplitudes = prop(source_amplitudes.to(device), x_s.to(device), x_r.to(device), dt)
    return receiver_amplitudes.cpu()


createdata = create_data


class PhySimulator(nn.Module):
    """Mini-batch Deepwave physical simulator for FWI/adversarial training loops.

    The class selects subsets of shots depending on the training status and then
    models the corresponding receiver gathers.
    """

    def __init__(
        self,
        dx,
        num_shots: int,
        num_batches: int,
        x_s: Tensor,
        x_r: Tensor,
        dt: float,
        pml_width,
        order: int,
        survey_pad,
    ) -> None:
        super().__init__()
        if num_shots % num_batches != 0:
            raise ValueError("num_shots must be divisible by num_batches.")
        self.dx = dx
        self.num_shots = num_shots
        self.num_batches = num_batches
        self.x_s = x_s
        self.x_r = x_r
        self.dt = dt
        self.num_shots_per_batch = num_shots // num_batches
        self.pml_width = pml_width
        self.order = order
        self.survey_pad = survey_pad

    def forward(
        self,
        model: Tensor,
        source_amplitudes: Tensor,
        it: int,
        critic_iter: int,
        j: int,
        status: str,
        add_noise: bool = False,
        noise_snr: Optional[float] = None,
        learn_awgn: bool = False,
    ) -> Tensor:
        """Run one mini-batch forward modeling pass.

        Parameters
        ----------
        model:
            Velocity model used by Deepwave.
        source_amplitudes:
            Source amplitudes for one shot/source configuration.
        it:
            Outer iteration index.
        critic_iter:
            Number of critic iterations used when ``status='TD'``.
        j:
            Inner critic iteration index.
        status:
            ``'TD'`` for discriminator/critic training or ``'TG'`` for generator/model training.
        add_noise:
            Whether to add AWGN to modeled data.
        noise_snr:
            Target SNR in dB if noise is added.
        learn_awgn:
            Backward-compatible flag; noise is added only if both this and ``add_noise`` are True.

        Returns
        -------
        Tensor
            Modeled receiver amplitudes for the selected batch.
        """
        _require_deepwave()
        prop = deepwave.scalar.Propagator({"vp": model}, self.dx, self.pml_width, self.order, self.survey_pad)
        batch_src_amps = source_amplitudes.repeat(1, self.num_shots_per_batch, 1)

        if status == "TD":
            batch_id = (it * critic_iter + j) % self.num_batches
        elif status == "TG":
            batch_id = it % self.num_batches
        else:
            raise ValueError("status must be 'TD' or 'TG'.")

        batch_x_s = self.x_s[batch_id::self.num_batches]
        batch_x_r = self.x_r[batch_id::self.num_batches]
        batch_data = prop(batch_src_amps, batch_x_s, batch_x_r, self.dt)

        if add_noise and noise_snr is not None and learn_awgn:
            batch_data = add_awgn(batch_data, noise_snr)
        return batch_data


# =============================================================================
# Loaders and model/source parameter utilities
# =============================================================================

def load_true_model(data_path: str, num_dims: int, vmodel_dim: np.ndarray) -> Tensor:
    """Load a 2D float32 binary velocity model.

    Parameters
    ----------
    data_path:
        Path to a raw float32 binary file.
    num_dims:
        Number of model dimensions. Currently only ``2`` is supported.
    vmodel_dim:
        Array-like dimensions ``[nx, nz]`` matching the original file layout.

    Returns
    -------
    Tensor
        Velocity model with shape ``[nz, nx]``.
    """
    if num_dims != len(vmodel_dim.reshape(-1)) or num_dims != 2:
        raise ValueError("Only 2D models are supported, and vmodel_dim must contain two values [nx, nz].")
    nx, nz = int(vmodel_dim[0]), int(vmodel_dim[1])
    model = np.fromfile(data_path, np.float32).reshape(nz, nx).T
    return torch.as_tensor(model, dtype=torch.float32)


loadtruemodel = load_true_model


def load_receiver_amplitudes(rcvfile: str, device: torch.device, key: str = "true") -> Tensor:
    """Load receiver amplitudes from a MATLAB ``.mat`` file."""
    data_mat = spio.loadmat(rcvfile)
    if key not in data_mat:
        raise KeyError(f"Key '{key}' not found in {rcvfile}.")
    return torch.as_tensor(data_mat[key], dtype=torch.float32, device=device)


loadrcv = load_receiver_amplitudes


def load_initial_model(initfile: str, device: torch.device, key: str = "initmodel") -> Tuple[Tensor, Tensor]:
    """Load an initial velocity model from a MATLAB ``.mat`` file."""
    model_mat = spio.loadmat(initfile)
    if key not in model_mat:
        raise KeyError(f"Key '{key}' not found in {initfile}.")
    model_init = torch.as_tensor(model_mat[key], dtype=torch.float32)
    model = model_init.clone().to(device).requires_grad_(True)
    return model, model_init


loadinitmodel = load_initial_model


def fix_model_grad(fix_value_depth: int, model: Tensor) -> None:
    """Register a gradient hook that freezes the shallow part of a velocity model.

    Gradients are set to zero for rows ``[:fix_value_depth, :]`` and preserved
    for rows ``[fix_value_depth:, :]``.
    """
    if fix_value_depth <= 0:
        raise ValueError("fix_value_depth must be positive.")
    gradient_mask = torch.zeros_like(model)
    gradient_mask[fix_value_depth:, :] = 1.0
    model.register_hook(lambda grad: grad.mul_(gradient_mask))


def load_initial_source(initsafile: str, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Load initial and true source wavelets from a MATLAB ``.mat`` file."""
    source_mat = spio.loadmat(initsafile)
    for key in ("initsource", "truesource"):
        if key not in source_mat:
            raise KeyError(f"Key '{key}' not found in {initsafile}.")
    source_init = torch.as_tensor(source_mat["initsource"], dtype=torch.float32, device=device)
    source_true = torch.as_tensor(source_mat["truesource"], dtype=torch.float32, device=device)
    return source_init, source_true


loadinitsource = load_initial_source


def create_learnable_snr(init_snr_guess: float, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create a learnable scalar SNR/noise parameter."""
    snr_init = torch.tensor(float(init_snr_guess), dtype=torch.float32)
    snr = snr_init.clone().to(device).requires_grad_(True)
    return snr, snr_init


createlearnSNR = create_learnable_snr


# =============================================================================
# Metrics: SSIM, SNR, RSNR, relative error, TV
# =============================================================================

def gaussian(window_size: int, sigma: float) -> Tensor:
    """Create a normalized 1D Gaussian kernel."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    x = torch.arange(window_size, dtype=torch.float32)
    center = window_size // 2
    gauss = torch.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Tensor:
    """Create a 2D Gaussian window used for SSIM computation."""
    one_d = gaussian(window_size, 1.5).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
    return two_d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(
    img1: Tensor,
    img2: Tensor,
    window: Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
    dynamic_range: float = 255.0,
) -> Tensor:
    """Internal SSIM implementation for 4D tensors [N, C, H, W]."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))


def compute_ssim(img1: ArrayLike, img2: ArrayLike, window_size: int = 11, size_average: bool = True) -> Tensor:
    """Compute structural similarity (SSIM) between two 2D or 3D arrays.

    For 2D inputs, data are treated as one image with one channel. For 3D inputs,
    the third dimension is treated as the batch/image index, matching the
    original implementation.
    """
    t1 = torch.as_tensor(_to_numpy(img1), dtype=torch.float32)
    t2 = torch.as_tensor(_to_numpy(img2), dtype=torch.float32)
    if t1.shape != t2.shape:
        raise ValueError(f"img1 and img2 must have the same shape, got {t1.shape} and {t2.shape}.")

    if t1.ndim == 2:
        t1 = t1[None, None, :, :]
        t2 = t2[None, None, :, :]
    elif t1.ndim == 3:
        t1 = t1.permute(2, 0, 1).unsqueeze(1)
        t2 = t2.permute(2, 0, 1).unsqueeze(1)
    else:
        raise ValueError("SSIM expects a 2D image or a 3D stack.")

    _, channel, _, _ = t1.shape
    window = create_window(window_size, channel).type_as(t1)
    return _ssim(t1, t2, window, window_size, channel, size_average)


ComputeSSIM = compute_ssim


def compute_snr(rec: ArrayLike, target: ArrayLike, eps: float = 1e-12) -> float:
    """Compute signal-to-noise ratio in dB between reconstruction and target."""
    rec_np = _to_numpy(rec)
    target_np = _to_numpy(target)
    if rec_np.shape != target_np.shape:
        raise ValueError("rec and target must have the same shape.")

    if rec_np.ndim == 3:
        snrs = []
        for i in range(rec_np.shape[0]):
            r = rec_np[i].ravel()
            t = target_np[i].ravel()
            snrs.append(10.0 * np.log10((np.sum(t ** 2) + eps) / (np.sum((r - t) ** 2) + eps)))
        return float(np.mean(snrs))
    if rec_np.ndim == 2:
        r = rec_np.ravel()
        t = target_np.ravel()
        return float(10.0 * np.log10((np.sum(t ** 2) + eps) / (np.sum((r - t) ** 2) + eps)))
    raise ValueError("rec and target must be 2D or 3D.")


ComputeSNR = compute_snr


def compute_rsnr(rec: ArrayLike, target: ArrayLike, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    """Compute regressed SNR and return the linearly corrected reconstruction."""
    rec_np = _to_numpy(rec)
    target_np = _to_numpy(target)
    if rec_np.shape != target_np.shape:
        raise ValueError("rec and target must have the same shape.")

    r = rec_np.ravel()
    t = target_np.ravel()
    slope, intercept, *_ = scipy.stats.linregress(r, t)
    corrected = slope * r + intercept
    rsnr = 10.0 * np.log10((np.sum(t ** 2) + eps) / (np.sum((corrected - t) ** 2) + eps))
    return float(rsnr), corrected.reshape(rec_np.shape)


ComputeRSNR = compute_rsnr


def compute_relative_error(rec: ArrayLike, target: ArrayLike, eps: float = 1e-12) -> float:
    """Compute relative L2 error ``||target-rec||_2 / ||target||_2``."""
    rec_np = _to_numpy(rec)
    target_np = _to_numpy(target)
    if rec_np.shape != target_np.shape:
        raise ValueError("rec and target must have the same shape.")
    return float(np.linalg.norm((target_np - rec_np).ravel()) / (np.linalg.norm(target_np.ravel()) + eps))


ComputeRE = compute_relative_error


def add_awgn(data: Tensor, snr: float) -> Tensor:
    """Add additive white Gaussian noise to receiver data at a target SNR.

    Parameters
    ----------
    data:
        Tensor with shape ``[nt, num_shots, num_receivers]``.
    snr:
        Target SNR in dB.
    """
    if data.ndim != 3:
        raise ValueError("data must have shape [nt, num_shots, num_receivers].")

    data_shot_first = data.permute(1, 0, 2)
    dim = data_shot_first.shape
    signal_power = torch.mean(data_shot_first.detach() ** 2, dim=(1, 2))
    noise_std = torch.sqrt(signal_power / (10.0 ** (snr / 10.0))).reshape(dim[0], 1, 1)
    noisy = data_shot_first + torch.randn_like(data_shot_first) * noise_std
    return noisy.permute(1, 0, 2)


AddAWGN = add_awgn


def tv_loss(x: Tensor) -> Tensor:
    """Compute isotropic total variation loss for a 2D image/model."""
    x = x.float()
    dh = (x[:, 1:] - x[:, :-1]).pow(2)
    dw = (x[1:, :] - x[:-1, :]).pow(2)
    return torch.sum(torch.sqrt(dh[:-1, :] + dw[:, :-1] + 1e-12)).float()


TVLoss = tv_loss


def anisotropic_tv_loss(x: Tensor) -> Tensor:
    """Compute L1 anisotropic total variation loss for a 2D image/model."""
    x = x.float()
    dh = x[:, 1:] - x[:, :-1]
    dw = x[1:, :] - x[:-1, :]
    return torch.sum(torch.abs(dh[:-1, :]) + torch.abs(dw[:, :-1])).float()


ATVLoss = anisotropic_tv_loss


# =============================================================================
# DIP/noise helpers and parameter selection
# =============================================================================

def update_input(
    net_input_saved: Tensor,
    noise: Tensor,
    iteration: int,
    reg_noise_std: float,
    reg_noise_decayevery: int,
) -> Tensor:
    """Add decaying random perturbation to a saved network input."""
    if reg_noise_decayevery != 0 and iteration % reg_noise_decayevery == 0:
        reg_noise_std *= 0.7
    return net_input_saved + noise.normal_() * reg_noise_std


updateinput = update_input


def fill_noise(x: Tensor, noise_type: str) -> None:
    """Fill a tensor in-place with uniform (``'u'``) or normal (``'n'``) noise."""
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        raise ValueError("noise_type must be 'u' for uniform or 'n' for normal.")


def np_to_torch(array: np.ndarray) -> Tensor:
    """Convert a NumPy array to a float32 torch tensor with a batch dimension."""
    return torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)


def get_noise(
    input_num: int,
    input_depth: int,
    method: str,
    spatial_size: Union[int, Tuple[int, int]],
    noise_type: str = "u",
    var: float = 1.0 / 10.0,
) -> Tensor:
    """Create a random noise tensor or normalized 2D meshgrid input.

    Returns shape ``[input_num, input_depth, H, W]`` for ``method='noise'`` and
    shape ``[1, 2, H, W]`` for ``method='meshgrid'``.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    h, w = spatial_size

    if method == "noise":
        net_input = torch.zeros([input_num, input_depth, h, w], dtype=torch.float32)
        fill_noise(net_input, noise_type)
        return net_input * var
    if method == "meshgrid":
        if input_depth != 2:
            raise ValueError("input_depth must be 2 when method='meshgrid'.")
        x, y = np.meshgrid(np.arange(w) / float(w - 1), np.arange(h) / float(h - 1))
        meshgrid = np.concatenate([x[None, :], y[None, :]], axis=0)
        return np_to_torch(meshgrid)
    raise ValueError("method must be 'noise' or 'meshgrid'.")


def get_params(opt_over: str, net: nn.Module, net_input: Tensor, downsampler: Optional[nn.Module] = None) -> list:
    """Return parameters to optimize based on a comma-separated option string.

    Examples
    --------
    ``opt_over='net'`` optimizes only network weights.
    ``opt_over='net,input'`` optimizes network weights and the input tensor.
    ``opt_over='down'`` optimizes a supplied downsampler module.
    """
    params = []
    for opt in opt_over.split(','):
        opt = opt.strip()
        if opt == "net":
            params += list(net.parameters())
        elif opt == "down":
            if downsampler is None:
                raise ValueError("downsampler must be provided when opt_over contains 'down'.")
            params += list(downsampler.parameters())
        elif opt == "input":
            net_input.requires_grad_(True)
            params.append(net_input)
        else:
            raise ValueError(f"Unknown optimization target: {opt}")
    return params


# =============================================================================
# Timing helpers
# =============================================================================

_TIC_START: Optional[float] = None


def tic() -> None:
    """Start a simple wall-clock timer."""
    global _TIC_START
    _TIC_START = time.time()


def toc() -> None:
    """Print elapsed time since ``tic`` was called."""
    if _TIC_START is None:
        print("Toc: start time not set")
    else:
        print(f"Elapsed time is {time.time() - _TIC_START:.6f} seconds.")


# =============================================================================
# Normalization and patch utilities
# =============================================================================

def normalize_to_range(arr: np.ndarray, target_min: float, target_max: float, eps: float = 1e-12) -> np.ndarray:
    """Linearly normalize an array to a target range."""
    arr = np.asarray(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return target_min + (arr - arr_min) * (target_max - target_min) / (arr_max - arr_min + eps)



def patch_torch(A: Tensor, l1: int, l2: int, o1: int, o2: int) -> Tensor:
    """Extract flattened overlapping 2D patches from a tensor.

    Parameters
    ----------
    A:
        Input tensor with shape ``[n1, n2]``.
    l1, l2:
        Patch size along each dimension.
    o1, o2:
        Patch stride along each dimension.

    Returns
    -------
    Tensor
        Patch matrix with shape ``[num_patches, l1*l2]``.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D [n1, n2], got {tuple(A.shape)}.")
    n1, n2 = A.shape
    device, dtype = A.device, A.dtype

    pad_rows = (o1 - (n1 - l1) % o1) % o1
    pad_cols = (o2 - (n2 - l2) % o2) % o2
    if pad_rows:
        A = torch.cat([A, torch.zeros((pad_rows, n2), device=device, dtype=dtype)], dim=0)
    if pad_cols:
        A = torch.cat([A, torch.zeros((A.shape[0], pad_cols), device=device, dtype=dtype)], dim=1)

    patches = A.unfold(0, l1, o1).unfold(1, l2, o2).contiguous()
    return patches.view(-1, l1 * l2)


def patch_inv_torch(X1: Tensor, n1: int, n2: int, l1: int, l2: int, o1: int, o2: int) -> Tensor:
    """Reconstruct a 2D tensor from flattened overlapping patches by averaging."""
    X = _standardize_patch_layout(X1, l1, l2)
    N1, N2 = _padded_size(n1, n2, l1, l2, o1, o2)
    A = torch.zeros((N1, N2), device=X.device, dtype=X.dtype)
    mask = torch.zeros_like(A)

    idx = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            patch = X[idx].view(l1, l2)
            A[i1:i1 + l1, i2:i2 + l2] += patch
            mask[i1:i1 + l1, i2:i2 + l2] += 1.0
            idx += 1
    return (A / mask.clamp_min(1e-12))[:n1, :n2]


def _standardize_patch_layout(X1: Tensor, l1: int, l2: int) -> Tensor:
    """Return patch tensor in shape ``[num_patches, l1*l2]``."""
    if X1.ndim != 2:
        raise ValueError(f"X1 must be 2D, got {tuple(X1.shape)}.")
    p = l1 * l2
    if X1.shape[0] == p:
        return X1.t().contiguous()
    if X1.shape[1] == p:
        return X1.contiguous()
    raise ValueError(f"One dimension of X1 must equal l1*l2={p}, got {tuple(X1.shape)}.")


def _padded_size(n1: int, n2: int, l1: int, l2: int, o1: int, o2: int) -> Tuple[int, int]:
    """Compute padded canvas size compatible with patch extraction."""
    pad1 = (o1 - (n1 - l1) % o1) % o1
    pad2 = (o2 - (n2 - l2) % o2) % o2
    return n1 + pad1, n2 + pad2


def gaussian_window_2d(
    l1: int,
    l2: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Create a peak-normalized separable 2D Gaussian blending window."""
    sigma1 = max(l1 / 6.0, 1.0) if sigma1 is None else sigma1
    sigma2 = max(l2 / 6.0, 1.0) if sigma2 is None else sigma2
    y = torch.arange(l1, device=device, dtype=dtype) - (l1 - 1) / 2.0
    x = torch.arange(l2, device=device, dtype=dtype) - (l2 - 1) / 2.0
    wy = torch.exp(-0.5 * (y / sigma1) ** 2)
    wx = torch.exp(-0.5 * (x / sigma2) ** 2)
    w2d = wy[:, None] * wx[None, :]
    return w2d / w2d.max().clamp_min(eps)


def patch_inv_torch_gaussian(
    X1: Tensor,
    n1: int,
    n2: int,
    l1: int,
    l2: int,
    o1: int,
    o2: int,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
) -> Tensor:
    """Reconstruct a 2D tensor from overlapping patches using Gaussian blending."""
    X = _standardize_patch_layout(X1, l1, l2)
    N1, N2 = _padded_size(n1, n2, l1, l2, o1, o2)
    A = torch.zeros((N1, N2), device=X.device, dtype=X.dtype)
    mask = torch.zeros_like(A)
    W = gaussian_window_2d(l1, l2, device=X.device, dtype=X.dtype, sigma1=sigma1, sigma2=sigma2)

    idx = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            patch = X[idx].view(l1, l2)
            A[i1:i1 + l1, i2:i2 + l2] += patch * W
            mask[i1:i1 + l1, i2:i2 + l2] += W
            idx += 1
    return (A / mask.clamp_min(1e-12))[:n1, :n2]


# =============================================================================
# Colormaps and plotting
# =============================================================================

SEAM_COLORS = [
    "#0016A8",  # deep blue
    "#005DDE",  # blue
    "#0EA5FF",  # cyan-blue
    "#7DB9E8",  # light blue
    "#D9E2EA",  # almost white
    "#F2E6DB",  # beige
    "#E6B98A",  # light orange
    "#D97A00",  # orange
]

seam_cmap = LinearSegmentedColormap.from_list("seam_velocity", SEAM_COLORS, N=256)


def plot_velocity_models(
    model_true: Tensor,
    model_init: Tensor,
    model_result: Tensor,
    dx: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    titles: Tuple[str, str, str] = ("True Model", "Initial Model", "DIIP Model"),
    cmap: Union[str, LinearSegmentedColormap] = "jet",
    figsize: Tuple[float, float] = (40, 20),
):
    """Plot true, initial, and recovered velocity models with a shared colorbar.

    This function replaces the original global-variable based ``plot`` function.
    It returns the Matplotlib figure and axes so the caller can save or modify them.
    """


    models = [model_true, model_init, model_result]
    arrays = [_to_numpy(m) for m in models]
    if vmin is None:
        vmin = min(float(np.nanmin(a)) for a in arrays)
    if vmax is None:
        vmax = max(float(np.nanmax(a)) for a in arrays)

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    im = None
    labels = ["a)", "b)", "c)"]
    for ax, arr, title, label in zip(axes, arrays, titles, labels):
        nz, nx = arr.shape
        extent = [0, nx * dx / 1000.0, nz * dx / 1000.0, 0]
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        ax.set_ylabel("Depth (km)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Position (km)", fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Velocity (m/s)", fontweight="bold")
    plt.show()
    return fig, axes



def normalize_vel(x,vmin,vmax):
    """
    Normalize a velocity model to the range [-1, 1].

    This function clips the velocity values to the predefined physical
    bounds `[vmin, vmax]` and then linearly rescales them into the range
    [-1, 1]. It is commonly used before feeding velocity models into
    neural networks, diffusion models, or other machine-learning
    frameworks that expect normalized inputs.

    Parameters
    ----------
    x : torch.Tensor
        Velocity model in physical units (m/s).

    Returns
    -------
    torch.Tensor
        Normalized velocity model with values in the range [-1, 1].

    Notes
    -----
    The normalization is performed as:

        v_clipped = clip(v, vmin, vmax)

        v_01 = (v_clipped - vmin) / (vmax - vmin)

        v_norm = 2 * v_01 - 1

    where:

    - `vmin` is the minimum allowed velocity.
    - `vmax` is the maximum allowed velocity.

    Values outside the physical range are clipped before scaling to
    prevent unrealistic model values from propagating through the
    learning pipeline.

    This function is the inverse of `denormalize_vel()`.

    Examples
    --------
    >>> v = torch.tensor([[1500., 2500., 4500.]])
    >>> v_norm = normalize_vel(v)

    >>> v_rec = denormalize_vel(v_norm)
    >>> torch.allclose(v, v_rec)
    True
    """
    x = torch.clamp(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)  # [0, 1]
    x = 2.0 * x - 1.0               # [-1, 1]
    return x

def denormalize_vel(x,vmin,vmax):
    """
    Convert a normalized velocity model from [-1, 1] back to the
    physical velocity range [vmin, vmax].

    This function is the inverse of `normalize_vel()`. It is typically
    used after a neural network predicts a normalized velocity model,
    allowing the result to be mapped back to real seismic velocities.

    Parameters
    ----------
    x : torch.Tensor
        Normalized velocity model with values approximately in [-1, 1].

    Returns
    -------
    torch.Tensor
        Velocity model in physical units (m/s), scaled to the range
        [vmin, vmax].

    Notes
    -----
    The transformation is:

        x_norm ? [-1, 1]
            
        x_01 = (x_norm + 1) / 2
            
        v = x_01 * (vmax - vmin) + vmin

    where `vmin` and `vmax` are global velocity bounds.
    """
    x = (x + 1.0) / 2.0
    x = x * (vmax - vmin) + vmin
    return x


def gaussian_kernel2d(
    kernel_size=11,
    sigma=2.0,
    device="cuda",
    dtype=torch.float32
):
    """
    Create a normalized 2D Gaussian convolution kernel.

    The kernel is commonly used for smoothing velocity models,
    regularization, low-pass filtering, or generating Gaussian-weighted
    blending masks.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square Gaussian kernel.
        Default is 11.

    sigma : float, optional
        Standard deviation of the Gaussian distribution in pixels.
        Larger values produce stronger smoothing.
        Default is 2.0.

    device : str or torch.device, optional
        Device where the kernel will be created.
        Default is "cuda".

    dtype : torch.dtype, optional
        Data type of the kernel.
        Default is torch.float32.

    Returns
    -------
    torch.Tensor
        Normalized Gaussian kernel of shape
        (kernel_size, kernel_size).

    Notes
    -----
    The kernel is generated as:

        G(x,y) = exp(-(x²+y²)/(2?²))

    and normalized so that:

        kernel.sum() = 1

    ensuring that convolution preserves the average amplitude.
    """
    ax = (
        torch.arange(kernel_size, device=device, dtype=dtype)
        - (kernel_size - 1) / 2.0
    )

    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    kernel = torch.exp(
        -(xx**2 + yy**2) / (2.0 * sigma**2)
    )

    kernel = kernel / kernel.sum()

    return kernel


def smooth_velocity_gaussian(
    v,
    kernel_size=11,
    sigma=2.0
):
    """
    Apply differentiable Gaussian smoothing to a velocity model.

    This function performs Gaussian low-pass filtering using
    PyTorch convolution operations, making it fully differentiable
    and suitable for optimization-based workflows such as:

    - Full-Waveform Inversion (FWI)
    - Diffusion Image Prior (DIIP)
    - Velocity-model regularization
    - Deep-learning-based inversion

    Parameters
    ----------
    v : torch.Tensor
        Velocity model of shape (H, W).

    kernel_size : int, optional
        Size of the Gaussian kernel.
        Larger kernels increase the smoothing footprint.
        Default is 11.

    sigma : float, optional
        Standard deviation of the Gaussian kernel.
        Larger values produce stronger smoothing.
        Default is 2.0.

    Returns
    -------
    torch.Tensor
        Smoothed velocity model with the same shape (H, W).

    Notes
    -----
    The implementation:

    1. Creates a normalized Gaussian kernel.
    2. Expands the velocity model to 4D format expected by Conv2D.
    3. Uses replicate padding to avoid edge artifacts.
    4. Applies Gaussian convolution.

    Since all operations are differentiable, gradients propagate
    naturally through the smoothing operation during backpropagation.

    Examples
    --------
    >>> v_smooth = smooth_velocity_gaussian(
    ...     velocity,
    ...     kernel_size=21,
    ...     sigma=5
    ... )

    >>> loss = criterion(v_smooth, target)
    >>> loss.backward()
    """
    device, dtype = v.device, v.dtype

    k = gaussian_kernel2d(
        kernel_size,
        sigma,
        device=device,
        dtype=dtype
    )

    k = k[None, None, :, :]  # (1,1,K,K)

    v4 = v[None, None, :, :]  # (1,1,H,W)

    pad = kernel_size // 2

    v4 = F.pad(
        v4,
        (pad, pad, pad, pad),
        mode="replicate"
    )

    out = F.conv2d(v4, k)

    return out[0, 0]

# Loss Functions
def rtm(syn, obs):
    loss = (syn*obs).mean()
    return loss

def l2(syn, obs):
    return ((syn - obs) ** 2).mean()

def cs(syn, obs, epsilon=1e-8):
    return torch.mean(1 - torch.nn.functional.cosine_similarity(syn.view(obs.shape), obs, dim=-1, eps=epsilon))

import torch
import torch.nn.functional as F
from typing import Tuple

def slope_estimate(
    d: torch.Tensor,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: int = 5,
    eps: float = 0.0,
    dips: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Local slope estimation using the Structure Tensor algorithm, implemented in PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Input dataset of size (n_z, n_x).
    dz : float, optional
        Sampling in z-axis (default: 1.0).
    dx : float, optional
        Sampling in x-axis (default: 1.0).
    smooth : int, optional
        Standard deviation for Gaussian kernel (default: 5).
    eps : float, optional
        Regularization term to avoid instability (default: 0.0).
    dips : bool, optional
        Return dips (True) instead of slopes (False).

    Returns
    -------
    slopes : torch.Tensor
        Estimated local slopes.
    anisotropies : torch.Tensor
        Estimated local anisotropies.
    """
    # Initialize tensors for slopes and anisotropies
    slopes = torch.zeros_like(d)
    anisos = torch.zeros_like(d)

    # Compute gradients
    gz, gx = torch.gradient(d, spacing=(dz, dx))
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # Apply Gaussian smoothing
    def gaussian_filter(tensor, sigma):
        kernel = torch.exp(-0.5 * torch.arange(-sigma, sigma + 1).square() / sigma**2)
        kernel = kernel / kernel.sum()
        kernel_2d = kernel[:, None] @ kernel[None, :]
        kernel_2d = kernel_2d.to(tensor.device, dtype=tensor.dtype)
        kernel_2d = kernel_2d.expand(1, 1, *kernel_2d.shape)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        return F.conv2d(tensor, kernel_2d, padding=sigma).squeeze()

    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    # Normalize gradients by maximum value
    gmax = max(gzz.max(), gxx.max(), torch.abs(gzx).max())
    if gmax <= eps:
        return slopes, anisos

    gzz = gzz/gmax.detach()
    gzx = gzx/gmax.detach()
    gxx = gxx/gmax.detach()

    # Compute eigenvalues of the smoothed tensor
    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * torch.sqrt((gzz - gxx)**2 + 4 * gzx**2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    # Regularization to avoid instability
    regdata = l1 > eps
    # anisos[regdata] = 1 - l2[regdata] / l1[regdata]
    anisos = 1 - l2 / l1

    if dips:
        slopes = 0.5 * torch.atan2(2 * gzx, gzz - gxx)
    else:
        regdata = torch.abs(gzx) > eps
        slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

    return slopes#, anisos

from scipy import signal
from torchaudio.functional import filtfilt

def decide_filter_type(freq):
    """Summary: Decide the filter type

    Args:
        freq (in, float, list): The frequency of the filter.

    Returns:
        str : The filter type.
    """
    if isinstance(freq, (int, float)): filter_mode = "lowpass"
    if isinstance(freq, (list, tuple)): 
        if len(freq) == 1: filter_mode = "lowpass"
        if len(freq) == 2: filter_mode = "bandpass"
    if freq == "all": filter_mode = "all"
    return filter_mode

def filter(d, freqs, dt=0.001, forder=3, btype=None, **kwargs):
    """Butterworth filter.

    Args:
        d (torch.Tensor): Data to be filtered. (nbatch, nsamples, ntraces, nchannels)
        freqs (int, float, list): The frequency of the filter.
        dt (float, optional): Sampling interval. Defaults to 0.001.
        forder (int, optional): Filter order. Defaults to 3.
        btype (str, optional): Filter type. Defaults to None, which will be decided by the function.
    """
    assert d.ndim == 4, f"The input data must be 4D (nbatch, nsamples, ntraces, nchannels)., but got {d.ndim}"
    
    filter_mode = decide_filter_type(freqs) if btype is None else btype

    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn
    b, a = signal.butter(forder, Wn=wn, btype=filter_mode)

    b = torch.from_numpy(b).double().to(d.device)
    a = torch.from_numpy(a).double().to(d.device)

    # Move the time axis to the last axis
    d = d.permute(0, 3, 2, 1)
    d = filtfilt(d.double(), a, b, clamp=False)
    d = d.float()
    # Move the time axis back to the original position
    d = d.permute(0, 3, 2, 1)
    return d


def filter_dw(d, freqs, dt=0.001, forder=3, btype=None, **kwargs):
    """Butterworth filter.

    Args:
        d (torch.Tensor): Data to be filtered. (..., nsamples)
        freqs (int, float, list): The frequency of the filter.
        dt (float, optional): Sampling interval. Defaults to 0.001.
        forder (int, optional): Filter order. Defaults to 3.
        btype (str, optional): Filter type. Defaults to None, which will be decided by the function.
    """
    
    filter_mode = decide_filter_type(freqs) if btype is None else btype

    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn
    b, a = signal.butter(forder, Wn=wn, btype=filter_mode)

    b = torch.from_numpy(b).double().to(d.device)
    a = torch.from_numpy(a).double().to(d.device)

    d = filtfilt(d.double(), a, b, clamp=False)
    return d.float()


import os
import torch
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class OBSDataset(Dataset):
    """The class for data loading.
       Supported file types: .npy, .hdf5
       For numpy files, we load the whole data into memory.
       For hdf5 files, we can load a single shot into memory.
    """

    def __init__(self, 
                 dpath, 
                 sources=None, 
                 receivers=None, 
                 Preload=True, 
                 endnt=None):

        self.dpath = dpath
        self.sources = sources
        self.receivers = receivers
        self.endnt = endnt
        if Preload:
            self.preload()

    def preload(self, ):
        self.data = np.load(self.dpath)[:, :self.endnt, ...]

    def __getitem__(self, shot_nos):
    
        src_this_epoch = self.sources[shot_nos]
        rec_this_epoch = self.receivers[shot_nos]

        data = self.data[shot_nos]
                
        return data, src_this_epoch, rec_this_epoch, shot_nos
    
    def __len__(self):
        return self.data.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Inversion with Implicit Nueral Representations")
    parser.add_argument("configure", type=str, help="Path to the configure file")
    parser.add_argument("--savepath", type=str, help="Path to save the inversion results")
    parser.add_argument("--ckptpath", type=str, default=None, help="Path to save the ckpt results (used for pretrain)")
    parser.add_argument("--use-hash", action="store_true", help="Use hash encoding for the coordiantes")
    parser.add_argument("--use-bias", action="store_true", help="Use bias in the network")
    parser.add_argument("--std", nargs='+', help='Model parameter key-value pairs, e.g. vp 300 vs 400')
    parser.add_argument("--mean", nargs='+', help='Model parameter key-value pairs, e.g. vp 300 vs 400')
    parser.add_argument("--lr", nargs='+', default=None, help='Learning rates for conventional inversion, e.g. vp 20.0 vs 15.0')
    parser.add_argument("--nnlr", default=None, help='Learning rates for neural network, e.g. 1e-3 1e-4')
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the network")
    parser.add_argument("--features", type=int, default=16, help="Number of features in the network")
    parser.add_argument("--use-multiscale", action="store_true", help="Use multiscale for the inversion")
    parser.add_argument("--use-ill", action="store_true", help="Use illumination for the inversion")
    parser.add_argument("--use-shot-ill", action="store_true", help="Use illumination for the inversion")
    parser.add_argument("--use-source-encoding", action="store_true", help="Use source encoding for the inversion")
    parser.add_argument("--use-dropout", action="store_true", help="Use dropout for uncertainty estimation")
    parser.add_argument("--illtype", type=str, default="s", choices=["s", "r", "sr"], help="Type of illumination, s for source, sr for source-residual")
    parser.add_argument("--loss", type=str, default="l2", choices=["l2", "cs", "rtm"], help="Type of loss function")
    parser.add_argument("--verbose", action="store_true", help="Print the verbose information")
    parser.add_argument("--filter-wavelet", action="store_true", help="Filter the wavelet at the beginning")
    parser.add_argument("--reset-state", action="store_true", help="Reset the state at the beginning of each frequency")
    parser.add_argument("--use-prev-as-init", action="store_true", help="Use results from previous frequency as the initial model")
    parser.add_argument("--model-split", action="store_true", help="Split the model into multiple parts for large models")
    parser.add_argument("--automesh", action="store_true", help="Automatically adjust the mesh (both dt and dx) size based on the model and frequency")
    parser.add_argument("--cal-odcig", action="store_true", help="Calculate the ODCIG (Offset Domain Common Image Gathers)")
    parser.add_argument("--use-vp-grad", action="store_true", help="Use vp gradient for all parameters")
    parser.add_argument("--pretrain-mlp-with-initial", action="store_true", help="Pretrain the model with initial models")
    parser.add_argument("--pretrain-unet-with-noise", action="store_true", help="Pretrain the unet with noise")
    parser.add_argument("--receiver-start-from-right", action="store_true", help="Receiver start from right")
    parser.add_argument("--cumsum", action="store_true", help="Integrate the wavelet")
    parser.add_argument("--grad-cut", action="store_true", help="Cut the gradient beyond the source and receiver range")
    parser.add_argument("--use-ddpm", action="store_true", help="Use DDPM for the inversion")
    

    return parser.parse_args()

# Ricker wavelet
def ricker(t, f=10.):
    """Ricker wavelet.

    Args:
        t (Array): Discrete time.
        f (float, optional): Dominant frequency. Defaults to 10..

    Returns:
        Array: Ricker wavelet series.
    """
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return r.astype(np.float32)

def initialize_wavelet(wavelet_cfg):
    """Generate or load the wavelet based on the configuration.

    Args:
        wavelet_cfg (dict): Configuration for the wavelet, which includes:
            - 'path': Path to the wavelet file (optional).
            - 'time': Total time duration of the wavelet.
            - 'dt': Time step of the wavelet.
            - 'delay': Delay time before the wavelet starts.
            - 'fm': Dominant frequency of the wavelet.
    If the path is provided and the file exists, it loads the wavelet from the file.
    If not, it generates a Ricker wavelet based on the provided parameters.

    Returns:
        Array: The generated or loaded wavelet.
    """
    wavelet_path = wavelet_cfg['path']
    if wavelet_path is not None and os.path.exists(wavelet_path):
        print(f"Load wavelet from {wavelet_path}")
        wave = np.load(wavelet_path)
    else:
        # Generate Ricker wavelet when the path is not provided or does not exist
        nt = wavelet_cfg['time']/wavelet_cfg['dt']
        t = np.arange(nt)*wavelet_cfg['dt'] - wavelet_cfg['delay']
        wave = np.array(ricker(t, f=wavelet_cfg['fm']), dtype=np.float32)
    return wave

def initialize_paths(save_root, pathlib=["code", "figures", "model", "data"]):
    """Initialize the paths for saving results.

    Args:
        save_root (str): The root path to save the results.
    """
    if save_root is not None and not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    path = {k: os.path.join(save_root, k) for k in pathlib}
    # Create directories if not exist
    for k, v in path.items():
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
    return path


def load_acquisition_data(configure, nx):
    # Construct the acquisition geometry
    # First check the path of sources and receivers, if available, load them
    source_path = configure['geom']['source_path']
    receiver_path = configure['geom']['receiver_path']

    if source_path is not None:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file {source_path} does not exist.")
        print(f'Loading sources from {source_path}')
        sources = np.load(source_path)
    else:
        scfg = configure['geom']['sources']
        print(f'Generating sources from configure file')
        # Make sure the source locations are within the domain
        end = min(nx, scfg['source_end'])
        end = end if end != -1 else nx
        sources_x = np.arange(scfg['source_start'], end, scfg['source_interval'])
        sources_z = np.ones_like(sources_x) * scfg['source_depth']
        sources = np.stack([sources_x, sources_z], axis=1)

    if receiver_path is not None:
        if not os.path.exists(receiver_path):
            raise FileNotFoundError(f"Receiver file {receiver_path} does not exist.")
        print(f'Loading receivers from {receiver_path}')
        receivers = np.load(receiver_path)
    else:
        rcfg = configure['geom']['receivers']
        print(f'Generating receivers from configure file')
        # Make sure the receiver locations are within the domain
        end = min(nx, rcfg['receiver_end'])
        end = end if end != -1 else nx
        receivers_x = np.arange(rcfg['receiver_start'], end, rcfg['receiver_interval'])
        receivers_z = np.ones_like(receivers_x) * rcfg['receiver_depth']
        receivers = np.stack([receivers_x, receivers_z], axis=1)
        # tile to (nshots, nreceivers, 2)
        receivers = np.tile(receivers[None, ...], (sources.shape[0], 1, 1))

    return sources, receivers


def show_gathers(rec, size=3, figsize=(8, 5), save_path=None, transpose=False):
    np.random.seed(0)
    randno = np.random.randint(0, rec.shape[0], size=size)
    fig,axes=plt.subplots(1, randno.shape[0], figsize=figsize)
    axes = [axes] if size == 1 else axes
    for i, ax in enumerate(axes):
        vmin,vmax=np.percentile(rec[i], [1, 99])
        kwargs=dict(vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        if transpose:
            ax.imshow(rec[randno[i]].T, **kwargs)
        else:
            ax.imshow(rec[randno[i]], **kwargs)
        ax.set_title(f"shot {randno[i]}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def show_frequency_spectrum(data, dt, end_freq, axis=0, savepath='./fs.png'):
    """Show the frequency spectrum of the data.

    Args:
        data (Array): The data to be analyzed.
        dt (float): Time step.
        end_freq (float): The end frequency to be displayed.
        axis (int, optional): Which axis to analyze. Defaults to 0.

    Returns:
        Array: The frequency spectrum (summed).
    """
    n = data.shape[axis]
    fft_data = np.fft.fft(data, axis=axis)
    freq_amp = np.abs(fft_data)
    # Sum the amplitude of the frequency except the axis
    freq_amp = np.sum(freq_amp, axis=tuple([i for i in range(data.ndim) if i != axis]))
    freq = np.fft.fftfreq(data.shape[axis], d=dt)
    # Only show the positive frequency
    freq_amp = freq_amp[freq>=0]
    freq = freq[freq>=0]
    # Only show the frequency less than the end_freq
    freq_amp = freq_amp[freq<=end_freq]
    freq = freq[freq<=end_freq]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(freq, freq_amp)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Frequency Spectrum")
    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def grid_init(shape, vmin=0., vmax=1., transpose=False):
    """Initialize the mesh of the domain

    Args:
        shape (tuple): The shape of the domain.
        vmin (float): The minimum value of the domain.
        vmax (float): The maximum value of the domain.

    Returns:
        Array : The mesh of the domain with shape (*shape, 2).
    """
    coord_axis = [np.linspace(vmin, vmax, d, dtype=np.float32) for d in shape]
    if not transpose:
        grid = np.stack(np.meshgrid(*coord_axis, indexing='ij'), -1)
    else:
        grid = np.stack(np.meshgrid(*coord_axis, indexing='xy'), -1)
    return grid.reshape(-1,2).astype(np.float32)

def parse_std_args(pair_args):
    if len(pair_args) % 2 != 0:
        raise ValueError("Arguments after --xxx must be in key-value pairs")
    return {
        pair_args[i]: float(pair_args[i+1])
        for i in range(0, len(pair_args), 2)
    }

def show_model(model, figsize=(5,3), save_path=None, **kwargs):
    """Show the model.

    Args:
        model (Array): The model to be displayed with shape (nz, nx).
        figsize (tuple, optional): The size of the figure. Defaults to (5,3).
        save_path (str, optional): The path to save the figure. Defaults to None.
    """
    fig,ax=plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    plt.colorbar(ax.imshow(model, **kwargs))
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def split_model(m, sources, receivers, one_side_expand=50, source_right=False):
    """Split the model based on the sources and receivers locations for saving memory and computational resources.

    Args:
        m (2d array): The model to be split, shape (nz, nx).
        sources (2d array): The source locations, shape (nshots, 2).
        receivers (3d array): The receiver locations, shape (nshots, nreceivers, 2).
        one_side_expand (int): The number of grids to expand for one side.

    Returns:
        tuple: A tuple containing:
            - m_split (array): The split velocity model, shape (nshots, nz, nx).
            - left (array): The leftmost index of the split model for each shot.
            - right (array): The rightmost index of the split model for each shot.
    """
    ori_domain = m[0].shape
    assert sources.shape[0] == receivers.shape[0], "Sources and receivers must have the same number of shots"
    assert sources.ndim == 3, "Sources must have shape (nshots, x, 2)"
    assert receivers.ndim == 3, "Receivers must have shape (nshots, nreceivers, 2)"

    m_split = []
    left = []
    right = []

    sources_moved = sources.copy()
    receivers_moved = receivers.copy()

    # Calculate the max length between the sources and receivers for each shot
    # and determine the leftmost and rightmost points
    # max_length = np.abs(np.max(receivers[..., 0], axis=1) - sources[..., 0]).max()
    max_length = np.max(np.abs(receivers[..., 0] - sources[..., 0]), axis=1).max()
    print('Max offset in x:', max_length)
    for shot in range(sources.shape[0]):
        srcx = sources[shot, 0, 0]
        recx = receivers[shot, ..., 0]

        leftmost_recx = np.min(recx)
        rightmost_recx = np.max(recx)

        leftmost = np.min(np.array([leftmost_recx, srcx]), axis=0)
        rightmost = np.max(np.array([rightmost_recx, srcx]), axis=0)

        # Expand the leftmost and rightmost points by the specified number of grids
        expand_left = one_side_expand

        if leftmost - expand_left < 0:
            left_start = 0
            expand_left = leftmost
            expand_right = one_side_expand*2 - expand_left
        else:
            left_start = leftmost - expand_left
            expand_right = one_side_expand

        right_end = left_start + expand_left + max_length + expand_right

        if right_end > ori_domain[1]: # if the right end exceeds the domain size, try to adjust

            right_end = ori_domain[1]
            expand_right = right_end - rightmost
            expand_left = one_side_expand * 2 - expand_right
            left_start = leftmost - expand_left
            assert left_start >= 0, f'Left start {left_start} is out of bounds for the domain size {ori_domain[1]}'
            assert right_end <= ori_domain[1], f'Right end {right_end} is out of bounds for the domain size {ori_domain[1]}'

        left_start = int(left_start)
        right_end = int(right_end)

        m_split.append([_m[:, left_start:right_end] for _m in m])
        left.append(left_start)
        right.append(right_end)

        sources_moved[shot, ..., 0] -= left_start
        receivers_moved[shot, ..., 0] -= left_start
    try:
        m_split = np.stack(m_split, axis=0)
    except ValueError as e:
        print(f"Error stacking m_split: {e}")
    left = np.array(left)
    right = np.array(right)

    return m_split, left, right, sources_moved, receivers_moved