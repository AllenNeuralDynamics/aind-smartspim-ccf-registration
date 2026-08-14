"""
Plot functions for easy and fast visualiztaion of images and
regsitration results
"""
import numpy as np
import matplotlib.pyplot as plt
import ants
from aind_anatomical_utils.ants_volume import regrid_axis_aligned_ants

# The canonical-LPS reference grid used to be built here, by measuring the
# world extent of the image and dividing by its spacing. Those two are indexed
# differently -- the extent along world axes, the spacing in the image's index
# order -- so they line up only when the image's index axes already match the
# world axes. Flips were fine; any permutation produced a wrong-shaped grid and
# a distorted plot. On the CCF annotation (axis-aligned, 944x1120x480) it built
# a 944x1009x534 reference rather than a shape-preserving one.
#
# regrid_axis_aligned_ants does this properly: it permutes and flips to the
# target orientation, so the result is invariant to the input's index order,
# and for an axis-aligned volume it is exact -- no interpolation, which also
# stops label volumes being blurred by the old interp_type="linear".


def _robust_limits(arr, pct=(1, 99), ignore_zeros=True):
    x = arr
    if ignore_zeros:
        x = x[x != 0]
    if x.size == 0:
        x = arr.reshape(-1)
    lo, hi = np.nanpercentile(x, pct)
    return float(lo), float(hi)


def plot_antsimgs(
    ants_img,
    figpath=None,
    title="",
    vmin=None,
    vmax=None,
    robust=True,
    pct=(1, 99),
    ignore_zeros=True,
    interpolation="linear",
    pad_mm=0.0,
    figsize=(18, 6),
    dpi=150,
):
    """
    Slicer-like display convention (radiological):
      - Axial/Coronal:  R on left, L on right
      - Axial:          A at top, P at bottom
      - Sagittal:       A on left, P on right
      - Superior up (coronal/sagittal)

    Internally regrids to canonical LPS, so the display is independent of the
    input's index order. Exact for axis-aligned volumes -- nothing is cropped,
    resampled or interpolated.
    """

    img = regrid_axis_aligned_ants(ants_img, "LPS")

    arr = img.numpy()
    sx, sy, sz = map(float, img.spacing)

    mid = (np.asarray(arr.shape) - 1) // 2
    i, j, k = mid.tolist()

    # Extract mid-slices, transpose so first axis is vertical for imshow
    sag = arr[i, :, :].T   # (Z, Y)
    cor = arr[:, j, :].T   # (Z, X)
    axi = arr[:, :, k].T   # (Y, X)

    Xmax = arr.shape[0] * sx
    Ymax = arr.shape[1] * sy
    Zmax = arr.shape[2] * sz

    # In LPS: +X=Left, +Y=Posterior, +Z=Superior.
    # To match Slicer:
    #   - R on left, L on right  => X should increase left->right? No:
    #       If +X is Left, then smaller X is more Right.
    #       To put R on the left side of the image, left side should be small X.
    #       So x-axis should run from Xmin to Xmax left->right (no reversal).
    #   - A at top, P at bottom  => Y should be small at top.
    #       Since +Y is Posterior, smaller Y is more Anterior.
    #       So y-axis should run from Ymax to 0 bottom->top? We accomplish A-up by reversing Y in extent.

    # Sagittal (YZ): x=Y should be A->P left->right; y=Z I->S bottom->top
    # Anterior corresponds to small Y, so no reverse for sagittal X extent.
    sag_extent = [0, Ymax, 0, Zmax]

    # Coronal (XZ): x=X should be R->L left->right; y=Z I->S bottom->top
    # Right corresponds to small X, so x=0 at left is correct => no reverse.
    cor_extent = [0, Xmax, 0, Zmax]

    # Axial (XY): x=X R->L left->right; y=Y should be A at top (small Y at top) => reverse Y extent
    axi_extent = [0, Xmax, Ymax, 0]

    if (vmin is None or vmax is None) and robust:
        lo, hi = _robust_limits(arr, pct=pct, ignore_zeros=ignore_zeros)
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    ax0.imshow(sag, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
               extent=sag_extent, aspect="equal")
    ax1.imshow(cor, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
               extent=cor_extent, aspect="equal")
    im = ax2.imshow(axi, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                    extent=axi_extent, aspect="equal")

    # Keep equal-aspect axes from being squeezed by layout
    def _set_box_aspect(ax, ext):
        w = abs(float(ext[1] - ext[0]))
        h = abs(float(ext[3] - ext[2]))
        ax.set_box_aspect(h / w)

    _set_box_aspect(ax0, sag_extent)
    _set_box_aspect(ax1, cor_extent)
    _set_box_aspect(ax2, axi_extent)

    ax0.set_title("Sagittal (X mid, LPS)", fontsize=12)
    ax1.set_title("Coronal (Y mid, LPS)", fontsize=12)
    ax2.set_title("Axial (Z mid, LPS)", fontsize=12)

    # Slicer-like labels
    ax0.set_xlabel("Anterior → Posterior (Y)")
    ax0.set_ylabel("Inferior → Superior (Z)")
    ax1.set_xlabel("Right → Left (X)")
    ax1.set_ylabel("Inferior → Superior (Z)")
    ax2.set_xlabel("Right → Left (X)")
    ax2.set_ylabel("Anterior → Posterior (Y)")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Intensity")

    if title:
        fig.suptitle(title, fontsize=14)

    if figpath:
        fig.savefig(figpath, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    return fig, (ax0, ax1, ax2)


def plot_reg(
    moving,
    fixed,
    warped,
    figpath=None,
    title="",
    loc=2,
    vmin=None,
    vmax=None,
    robust=True,
    pct=(1, 99),
    ignore_zeros=True,
    interpolation="linear",
    diff_scale="robust",   # "robust" or "match_vmax"
    pad_mm=0.0,
    figsize=None,          # if None, auto from physical extent
    max_inches=24,
    dpi=150,
):
    """
    Registration QC plot matching 3D Slicer (radiological) conventions.

    Workflow:
      1) Resample moving/fixed/warped into an *enclosing* canonical LPS grid
         (identity direction) to avoid cropping.
      2) Plot geometry in mm using extent + aspect='equal'.
      3) Display convention matches Slicer slice viewers:
         - Coronal/Axial: Right on left, Left on right  (R→L increases left→right)
         - Axial: Anterior at top, Posterior at bottom (A→P increases top→bottom)
         - Sagittal: Anterior on left, Posterior on right (A→P left→right)
         - Superior up where applicable

    Parameters
    ----------
    loc : int
        0=sagittal (X mid), 1=coronal (Y mid), 2=axial (Z mid) after LPS reorientation.
    """

    if loc not in (0, 1, 2):
        raise ValueError("loc must be 0, 1, or 2.")

    # --- Canonical LPS ---
    # Regrid `fixed`, then bring the other two onto it, so all three still share
    # one grid for the overlay while the grid itself no longer depends on
    # `fixed`'s index order.
    fixed_lps = regrid_axis_aligned_ants(fixed, "LPS")
    moving_lps = ants.resample_image_to_target(
        moving, fixed_lps, interp_type=interpolation
    )
    warped_lps = ants.resample_image_to_target(
        warped, fixed_lps, interp_type=interpolation
    )

    mov = moving_lps.numpy()
    fix = fixed_lps.numpy()
    wrp = warped_lps.numpy()

    sx, sy, sz = map(float, fixed_lps.spacing)

    mid = (np.asarray(fix.shape) - 1) // 2
    i, j, k = mid.tolist()

    # Slices returned with first axis = vertical for imshow
    if loc == 0:
        # sagittal: (Z, Y)
        mov2 = mov[i, :, :].T
        fix2 = fix[i, :, :].T
        wrp2 = wrp[i, :, :].T
        panel_w = fix.shape[1] * sy  # Y
        panel_h = fix.shape[2] * sz  # Z
        extent = [0, panel_w, 0, panel_h]  # x=Y, y=Z
        xlabel, ylabel = "Anterior → Posterior (Y)", "Inferior → Superior (Z)"
        view_title = "Sagittal (X mid, LPS)"
    elif loc == 1:
        # coronal: (Z, X)
        mov2 = mov[:, j, :].T
        fix2 = fix[:, j, :].T
        wrp2 = wrp[:, j, :].T
        panel_w = fix.shape[0] * sx  # X
        panel_h = fix.shape[2] * sz  # Z
        extent = [0, panel_w, 0, panel_h]  # x=X, y=Z
        xlabel, ylabel = "Right → Left (X)", "Inferior → Superior (Z)"
        view_title = "Coronal (Y mid, LPS)"
    else:
        # axial: (Y, X)
        mov2 = mov[:, :, k].T
        fix2 = fix[:, :, k].T
        wrp2 = wrp[:, :, k].T
        panel_w = fix.shape[0] * sx  # X
        panel_h = fix.shape[1] * sy  # Y
        # Reverse Y extent so anterior (small Y) appears at the top (Slicer-like)
        extent = [0, panel_w, panel_h, 0]  # x=X, y=Y reversed
        xlabel, ylabel = "Right → Left (X)", "Anterior → Posterior (Y)"
        view_title = "Axial (Z mid, LPS)"

    # Intensity range from fixed slice
    if (vmin is None or vmax is None) and robust:
        lo, hi = _robust_limits(fix2, pct=pct, ignore_zeros=ignore_zeros)
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi

    overlay = np.stack((wrp2, fix2, wrp2), axis=2)
    diff = fix2 - wrp2

    if diff_scale == "match_vmax":
        diff_vmin, diff_vmax = -float(vmax), float(vmax)
    elif diff_scale == "robust":
        dv = np.nanpercentile(np.abs(diff), 99)
        diff_vmin, diff_vmax = -float(dv), float(dv)
    else:
        raise ValueError("diff_scale must be 'robust' or 'match_vmax'.")

    # --- Figure sizing (auto, respects physical aspect) ---
    if figsize is None:
        base_h_in = 5.0
        total_w_in = base_h_in * (5 * (panel_w / panel_h)) + 1.2  # 5 panels + cbar
        if total_w_in > max_inches:
            scale = max_inches / total_w_in
            total_w_in = max_inches
            base_h_in *= scale
        figsize = (total_w_in, base_h_in)

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Use physical widths to allocate columns; normalize to avoid huge ratio values
    w = float(panel_w)
    gs = fig.add_gridspec(1, 6, width_ratios=[w, w, w, w, w, 0.06 * w])

    axes = [fig.add_subplot(gs[0, c]) for c in range(5)]
    cax = fig.add_subplot(gs[0, 5])

    axes[0].imshow(mov2, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                   extent=extent, aspect="equal")
    axes[1].imshow(fix2, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                   extent=extent, aspect="equal")
    im = axes[2].imshow(wrp2, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                        extent=extent, aspect="equal")
    axes[3].imshow(overlay, origin="lower", extent=extent, aspect="equal")
    axes[4].imshow(diff, cmap="gray", vmin=diff_vmin, vmax=diff_vmax, origin="lower",
                   extent=extent, aspect="equal")

    # Ensure layout doesn't squeeze equal-aspect axes
    def _set_box_aspect(ax):
        ax.set_box_aspect(panel_h / panel_w)

    for ax in axes:
        _set_box_aspect(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    titles = ["Moving", "Fixed", "Warped", "Warped ⊕ Fixed", "Fixed − Warped"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=12)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Intensity")

    if title:
        fig.suptitle(f"{title}\n{view_title}", fontsize=14)
    else:
        fig.suptitle(view_title, fontsize=14)

    if figpath:
        fig.savefig(figpath, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    return fig, axes
# def plot_antsimgs(ants_img, figpath, title="", vmin=0, vmax=500):
#     """
#     Plot ANTs image

#     Parameters
#     ------------
#     ants_img: ANTsImage
#     figpath: PathLike
#         Path where the plot is going to be saved
#     title: str
#         Figure title
#     vmin: float
#         Set the color limits of the current image.
#     vmax: float
#         Set the color limits of the current image.
#     """

#     if figpath:
#         ants_img = ants_img.numpy()
#         half_size = np.array(ants_img.shape) // 2
#         fig, ax = plt.subplots(1, 3, figsize=(10, 6))
#         ax[0].imshow(
#             ants_img[half_size[0], :, :], cmap="gray", vmin=vmin, vmax=vmax
#         )
#         ax[1].imshow(
#             ants_img[:, half_size[1], :], cmap="gray", vmin=vmin, vmax=vmax
#         )
#         im = ax[2].imshow(
#             ants_img[
#                 :,
#                 :,
#                 half_size[2],
#             ],
#             cmap="gray",
#             vmin=vmin,
#             vmax=vmax,
#         )
#         fig.suptitle(title, y=0.9)
#         plt.colorbar(
#             im, ax=ax.ravel().tolist(), fraction=0.1, pad=0.025, shrink=0.7
#         )
#         plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1)

# def plot_reg(
#     moving, fixed, warped, figpath, title="", loc=0, vmin=0, vmax=1.5
# ):
#     """
#     Plot registration results: moving, fixed, deformed,
#     overlay and difference images after registration

#     Parameters
#     ------------
#     moving: ANTsImage
#         Moving image
#     fixed: ANTsImage
#         Fixed image
#     warped: ANTsImage
#         Deformed image
#     figpath: PathLike
#         Path where the plot is going to be saved
#     title: str
#         Figure title
#     loc: int
#         Visualization direction
#     vmin, vmax: float
#         Set the color limits of the current image.
#     """

#     if loc >= len(moving.shape):
#         raise ValueError(
#             f"loc {loc} is not allowed, should less than {len(moving.shape)}"
#         )

#     half_size_moving = np.array(moving.shape) // 2
#     half_size_fixed = np.array(fixed.shape) // 2
#     half_size_warped = np.array(warped.shape) // 2

#     if loc == 0:
#         moving = moving.view()[half_size_moving[0], :, :]
#         fixed = fixed.view()[half_size_fixed[0], :, :]
#         warped = warped.view()[half_size_warped[0], :, :]
#         y = 0.75
#     elif loc == 1:
#         # moving = np.rot90(moving.view()[:,half_size[1], :], 3)
#         moving = moving.view()[:, half_size_moving[1], :]
#         fixed = fixed.view()[:, half_size_fixed[1], :]
#         warped = warped.view()[:, half_size_warped[1], :]
#         y = 0.82
#     elif loc == 2:
#         moving = np.rot90(np.fliplr(moving.view()[:, :, half_size_moving[2]]))
#         fixed = np.rot90(np.fliplr(fixed.view()[:, :, half_size_fixed[2]]))
#         warped = np.rot90(np.fliplr(warped.view()[:, :, half_size_warped[2]]))
#         y = 0.82
#     else:
#         raise ValueError(
#             f"loc {loc} is not allowed. Allowed values are: 0, 1, 2"
#         )

#     # combine deformed and fixed images to an RGB image
#     overlay = np.stack((warped, fixed, warped), axis=2)
#     diff = fixed - warped

#     fontsize = 14

#     fig, ax = plt.subplots(1, 5, figsize=(16, 6))
#     ax[0].imshow(moving, cmap="gray", vmin=vmin, vmax=vmax)
#     ax[1].imshow(fixed, cmap="gray", vmin=vmin, vmax=vmax)
#     ax[2].imshow(warped, cmap="gray", vmin=vmin, vmax=vmax)
#     ax[3].imshow(overlay)
#     ax[4].imshow(diff, cmap="gray", vmin=-(vmax), vmax=vmax)

#     ax[0].set_title("Moving", fontsize=fontsize)
#     ax[1].set_title("Fixed", fontsize=fontsize)
#     ax[2].set_title("Deformed", fontsize=fontsize)
#     ax[3].set_title("Deformed Overlay Fixed", fontsize=fontsize)
#     ax[4].set_title("Fixed - Deformed", fontsize=fontsize)

#     fig.suptitle(title, size=18, y=y)

#     if figpath:
#         plt.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
#         plt.close()
#     else:
#         fig.show()