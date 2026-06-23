"""Synthetic crack dataset generator.

Generates realistic-looking concrete crack images on-the-fly with:
  - Concrete texture background (smooth noise)
  - Jagged crack lines via Bezier/Catmull-Rom curves
  - Varying crack width (1-5 px)
  - Data augmentation (rotation, grid distortion, brightness)
  - Ground-truth masks, boundary maps, and direction maps

Matches the paper's dataset: 1850 train + 950 test at 320x320.
"""

import math
import random
import numpy as np
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _catrom_to_bezier(pts: np.ndarray) -> np.ndarray:
    """Convert Catmull-Rom control points to Bezier control points.

    Args:
        pts: (N, 2) control points (N >= 2)
    Returns:
        (N_segments * 4, 2) Bezier control points (4 per segment)
    """
    n = len(pts)
    if n < 2:
        return pts
    if n == 2:
        # Linear case: degenerate Bezier
        p0, p1 = pts[0], pts[1]
        return np.array([p0, p0, p1, p1])

    segments = []
    for i in range(n - 1):
        p0 = pts[max(0, i - 1)]
        p1 = pts[i]
        p2 = pts[min(n - 1, i + 1)]
        p3 = pts[min(n - 1, i + 2)]

        # Catmull-Rom to Bezier
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2

        segments.extend([b0, b1, b2, b3])

    return np.array(segments)


def _eval_bezier(control_pts: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Evaluate a cubic Bezier curve at num_points along t in [0, 1].

    Args:
        control_pts: (N, 2) where N is a multiple of 4
        num_points: number of sample points
    Returns:
        (num_points, 2) curve points
    """
    n_segments = len(control_pts) // 4
    points_per_segment = num_points // n_segments

    all_pts = []
    for i in range(n_segments):
        b0 = control_pts[4 * i]
        b1 = control_pts[4 * i + 1]
        b2 = control_pts[4 * i + 2]
        b3 = control_pts[4 * i + 3]

        ts = np.linspace(0, 1, points_per_segment)
        # Cubic Bezier: (1-t)^3 * P0 + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * P3
        pts = (
            (1 - ts[:, None]) ** 3 * b0[None, :]
            + 3 * (1 - ts[:, None]) ** 2 * ts[:, None] * b1[None, :]
            + 3 * (1 - ts[:, None]) * ts[:, None] ** 2 * b2[None, :]
            + ts[:, None] ** 3 * b3[None, :]
        )
        all_pts.append(pts)

    return np.concatenate(all_pts, axis=0)


def _draw_crack(mask: np.ndarray, points: np.ndarray,
                width: float = 2.0) -> np.ndarray:
    """Draw a crack line on the mask with variable width.

    Args:
        mask: (H, W) float32, updated in-place
        points: (N, 2) curve points in (x, y)
        width: base crack width in pixels
    Returns:
        Updated mask
    """
    H, W = mask.shape
    n = len(points)

    for i in range(n - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]

        # Skip if out of bounds
        if not (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
            continue

        # Compute local width (vary along the crack)
        local_width = width * (0.5 + 0.5 * math.sin(i / n * math.pi * 3))
        r = max(1, int(local_width))

        # Draw line segment with anti-aliasing
        xx, yy = np.linspace(x0, x1, int(np.hypot(x1 - x0, y1 - y0)) * 2 + 1), \
                 np.linspace(y0, y1, int(np.hypot(x1 - x0, y1 - y0)) * 2 + 1)

        for x, y in zip(xx, yy):
            xi, yi = int(round(x)), int(round(y))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        px, py = xi + dx, yi + dy
                        if 0 <= px < W and 0 <= py < H:
                            # Gaussian falloff for anti-aliasing
                            intensity = math.exp(
                                -(dx * dx + dy * dy) / (2 * r * r * 0.5 + 1)
                            )
                            mask[py, px] = max(mask[py, px], intensity)

    return mask


def _generate_concrete_background(
    size: Tuple[int, int], rng: np.random.RandomState
) -> np.ndarray:
    """Generate a concrete-like texture background.

    Args:
        size: (H, W)
        rng: random state
    Returns:
        (H, W) float32 grayscale image [0, 1]
    """
    H, W = size

    # Base noise
    noise = rng.randn(H, W)

    # Smooth with Gaussian filter
    from scipy.ndimage import gaussian_filter
    sigma = rng.uniform(1.5, 3.0)
    smooth = gaussian_filter(noise, sigma=sigma)

    # Normalize to [0, 1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)

    # Map to concrete-like grayscale values (120-180 on 0-255)
    base_color = rng.randint(100, 200)
    color_range = rng.randint(20, 60)
    image = base_color / 255.0 + color_range / 255.0 * (smooth - 0.5)

    # Add some darker spots (aggregate)
    n_spots = rng.randint(5, 20)
    for _ in range(n_spots):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        spot_r = rng.randint(3, 15)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        spot_mask = dist < spot_r
        image[spot_mask] = np.clip(
            image[spot_mask] - rng.uniform(0.02, 0.08), 0, 1
        )

    return np.clip(image, 0, 1).astype(np.float32)


def _generate_single_crack(
    size: Tuple[int, int],
    rng: np.random.RandomState,
    max_points: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single crack curve and its masks.

    Args:
        size: (H, W)
        rng: random state
        max_points: max control points for the curve
    Returns:
        mask: (H, W) float32 crack mask
        boundary: (H, W) float32 boundary edge map
        direction: (H, W) uint8 direction map (0-7, 0=no crack)
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.float32)

    # Random control points
    n_pts = rng.randint(3, max_points + 1)
    pts = np.zeros((n_pts, 2))
    pts[:, 0] = rng.randint(int(W * 0.05), int(W * 0.95), size=n_pts)
    pts[:, 1] = rng.randint(int(H * 0.05), int(H * 0.95), size=n_pts)

    # Sort by x to create a left-to-right crack
    pts = pts[pts[:, 0].argsort()]

    # Convert to Bezier and evaluate
    bezier_pts = _catrom_to_bezier(pts)
    curve = _eval_bezier(bezier_pts, num_points=200)

    # Random width
    width = rng.uniform(0.5, 3.0)

    # Draw the crack
    mask = _draw_crack(mask, curve, width=width)

    # Generate boundary (edges of the crack)
    from scipy.ndimage import binary_dilation, binary_erosion
    binary = (mask > 0.3).astype(np.float32)
    eroded = binary_erosion(binary, iterations=1)
    boundary = np.clip(binary - eroded, 0, 1).astype(np.float32)

    # Generate direction map (8 directions, 45 degrees each)
    direction = np.zeros((H, W), dtype=np.uint8)
    crack_pixels = binary > 0.5
    if crack_pixels.any():
        ys, xs = np.where(crack_pixels)
        for i in range(len(ys)):
            # Find local gradient direction
            y, x = ys[i], xs[i]
            # Look at neighborhood
            y1 = max(0, y - 2)
            y2 = min(H, y + 3)
            x1 = max(0, x - 2)
            x2 = min(W, x + 3)
            patch = binary[y1:y2, x1:x2]
            if patch.sum() < 5:
                continue
            # Compute principal direction via PCA on coordinates
            coords = np.array(np.where(patch)).T  # (N, 2)
            coords = coords - coords.mean(axis=0)
            if len(coords) < 5:
                continue
            cov = coords.T @ coords
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_dir = eigvecs[:, np.argmax(eigvals)]
            angle = math.degrees(math.atan2(main_dir[0], main_dir[1]))
            if angle < 0:
                angle += 360
            # Quantize to 8 directions (0: 0 deg, 1: 45 deg, ..., 7: 315 deg)
            dir_idx = int(round(angle / 45)) % 8
            direction[y, x] = dir_idx + 1  # 1-8 (0 = no crack)

    return mask, boundary, direction


class SyntheticCrackDataset(Dataset):
    """On-the-fly synthetic crack dataset.

    Generates concrete-like images with realistic cracks.
    Matches the paper's configuration:
    - 1850 training samples
    - 950 test samples
    - 320x320 resolution
    """

    CRACK_TYPES = ["single", "forked", "network", "curve"]

    def __init__(
        self,
        num_samples: int = 1850,
        size: Tuple[int, int] = (320, 320),
        split: str = "train",
        augment: bool = True,
        seed: int = 42,
        return_direction: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.size = size
        self.split = split
        self.augment = augment and split == "train"
        self.return_direction = return_direction
        self.base_seed = seed + (0 if split == "train" else 10000)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(self.base_seed + idx)

        # Generate concrete background (2D grayscale)
        image_gray = _generate_concrete_background(self.size, rng)

        # Convert grayscale to 3-channel RGB
        image = np.stack([image_gray] * 3, axis=-1)

        # Generate 1-4 cracks
        mask = np.zeros(self.size, dtype=np.float32)
        boundary = np.zeros(self.size, dtype=np.float32)
        direction = np.zeros(self.size, dtype=np.uint8)

        n_cracks = rng.randint(1, 5)
        for _ in range(n_cracks):
            c_mask, c_boundary, c_dir = _generate_single_crack(
                self.size, rng
            )
            mask = np.maximum(mask, c_mask)
            boundary = np.maximum(boundary, c_boundary)
            # Merge direction maps
            dir_mask = c_dir > 0
            direction[dir_mask] = c_dir[dir_mask]

        # Apply crack to image (darker pixels)
        crack_intensity = rng.uniform(0.3, 0.7)
        for c in range(3):
            image[:, :, c] = np.where(
                mask > 0.3,
                image[:, :, c] * (1.0 - crack_intensity),
                image[:, :, c],
            )

        # Binary mask
        mask_bin = (mask > 0.3).astype(np.float32)

        # Augmentation
        if self.augment:
            image, mask_bin, boundary, direction = self._augment(
                image, mask_bin, boundary, direction, rng
            )

        # Convert to torch tensors
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()  # (3, H, W)
        mask_bin = torch.from_numpy(mask_bin).float().unsqueeze(0)  # (1, H, W)
        boundary = torch.from_numpy(boundary).float().unsqueeze(0)  # (1, H, W)
        direction = torch.from_numpy(direction).long()  # (H, W)

        result = {
            "image": image,
            "mask": mask_bin,
            "boundary": boundary,
        }
        if self.return_direction:
            result["direction"] = direction

        return result

    def _augment(self, image, mask, boundary, direction, rng):
        """Apply data augmentation matching the paper:
        - Random rotation (-90 to 90 degrees)
        - 5x5 grid distortion
        - Random brightness/contrast
        """
        H, W = self.size

        # Random rotation
        if rng.random() < 0.8:
            angle = rng.uniform(-90, 90)
            from scipy.ndimage import rotate as ndi_rotate

            # Rotate image (RGB)
            rotated = np.zeros_like(image)
            for c in range(3):
                rotated[:, :, c] = ndi_rotate(
                    image[:, :, c], angle, reshape=False, order=1
                )
            image = rotated
            mask = ndi_rotate(mask, angle, reshape=False, order=0)
            boundary = ndi_rotate(boundary, angle, reshape=False, order=0)
            # Direction needs careful handling; skip rotation for direction

        # Grid distortion (simplified with random elastic deformation)
        if rng.random() < 0.5:
            from scipy.ndimage import map_coordinates

            sigma = rng.uniform(5, 10)
            alpha = rng.uniform(10, 30)

            # Random displacement fields
            dx = rng.uniform(-alpha, alpha, size=(5, 5))
            dy = rng.uniform(-alpha, alpha, size=(5, 5))

            # Interpolate to full resolution
            x = np.linspace(0, W - 1, 5)
            y = np.linspace(0, H - 1, 5)
            x_full = np.arange(W)
            y_full = np.arange(H)

            from scipy.interpolate import RectBivariateSpline
            interp_x = RectBivariateSpline(y, x, dx, kx=1, ky=1)
            interp_y = RectBivariateSpline(y, x, dy, kx=1, ky=1)
            full_dx = interp_x(y_full, x_full)
            full_dy = interp_y(y_full, x_full)

            # Apply deformation
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            coords = np.stack([yy + full_dy, xx + full_dx], axis=0)

            for c in range(3):
                image[:, :, c] = map_coordinates(
                    image[:, :, c], coords, order=1, mode="constant"
                )
            mask = map_coordinates(mask, coords, order=0, mode="constant")
            boundary = map_coordinates(
                boundary, coords, order=0, mode="constant"
            )

        # Random brightness
        if rng.random() < 0.5:
            factor = rng.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 1)

        return image, mask, boundary, direction


def get_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from config.

    Args:
        config: configuration dict with keys:
            data.num_train, data.num_test, data.input_size,
            data.batch_size, data.num_workers
    Returns:
        (train_loader, test_loader)
    """
    data_cfg = config["data"]

    train_dataset = SyntheticCrackDataset(
        num_samples=data_cfg["num_train"],
        size=(data_cfg["input_size"], data_cfg["input_size"]),
        split="train",
        augment=True,
        return_direction=config.get("training", {}).get("ibr", {}).get(
            "enabled", False
        ),
    )

    test_dataset = SyntheticCrackDataset(
        num_samples=data_cfg["num_test"],
        size=(data_cfg["input_size"], data_cfg["input_size"]),
        split="test",
        augment=False,
        return_direction=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader