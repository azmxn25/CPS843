# ============================================================
# Assignment 2 - Unsharp Masking and High-Boost Filtering and Canny Edge Detection
# Name: Azman Nazim
# Date: 2026-10-19
# ============================================================
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_image.png"
OUTPUT_DIR = BASE_DIR / "output"
# Part 1: Gaussian blur settings
BLUR_KERNEL_SIZE = 5       # Must be an odd positive integer
BLUR_SIGMA = 1.0
# Part 2: Canny settings. We blur the image ourselves so that sigma is
# explicitly controlled, then apply OpenCV's built-in Canny implementation.
CANNY_CONFIGS = [
    {"name": "Small smoothing, thresholds 50/150",
     "sigma": 0.8, "low": 50, "high": 150},
    {"name": "Large smoothing, thresholds 50/150",
     "sigma": 2.0, "low": 50, "high": 150},
    {"name": "Small smoothing, thresholds 100/200",
     "sigma": 0.8, "low": 100, "high": 200},
    {"name": "Small smoothing, thresholds 20/60",
     "sigma": 0.8, "low": 20, "high": 60},
    {"name": "Large smoothing, thresholds 100/200",
     "sigma": 2.0, "low": 100, "high": 200},
    {"name": "Large smoothing, thresholds 20/60",
     "sigma": 2.0, "low": 20, "high": 60},
]
def load_grayscale_image(path: Path) -> np.ndarray:
    """Read an image from disk and return an 8-bit grayscale image."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"Could not read '{path}'. Put input_image.png next to this script "
            "or update INPUT_PATH at the top of the file."
        )
    return image
def save_grayscale(path: Path, image: np.ndarray) -> None:
    """Save a grayscale image and check that the save succeeded."""
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"Could not save image to '{path}'")
def part1_unsharp_and_highboost(image: np.ndarray) -> None:
    """
    Implement unsharp masking/high-boost step by step:
      A. Gaussian blur
      B. Subtract blur from original to form the mask
      C. Add k times the mask back to the original
    """
    print("\nPART 1: Unsharp masking and high-boost filtering")
    # Convert uint8 values to float32 before subtraction and multiplication.
    # This allows negative mask values and prevents uint8 wrap-around.
    original = image.astype(np.float32)
    # Step A: create a low-frequency/smoothed version of the image.
    blurred = cv2.GaussianBlur(
        original,
        (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE),
        sigmaX=BLUR_SIGMA
    )
    # Step B: subtract the blurred image from the original.
    # The mask contains positive and negative values, especially near edges.
    unsharp_mask = original - blurred
    # The mask has negative values, so shift it to mid-gray (128) for display.
    # A gain of 10 makes the faint edge responses visible.
    MASK_GAIN = 10
    mask_display = np.clip(128 + MASK_GAIN * unsharp_mask, 0, 255).astype(np.uint8)
    # Step C: sharpen using g = f + k * mask.
    # k=1 is standard unsharp masking; k>1 is high-boost filtering.
    sharpened = {}
    for k in (1.0, 2.0, 4.5):
        result = original + k * unsharp_mask
        # Clip before converting to uint8 to avoid overflow/underflow artifacts.
        result = np.clip(result, 0, 255).astype(np.uint8)
        sharpened[k] = result
    # Save the intermediate results and final results.
    save_grayscale(OUTPUT_DIR / "part1_blurred.png",
                   np.clip(blurred, 0, 255).astype(np.uint8))
    save_grayscale(OUTPUT_DIR / "part1_unsharp_mask_display.png", mask_display)
    save_grayscale(OUTPUT_DIR / "part1_sharpened_k1.png", sharpened[1.0])
    save_grayscale(OUTPUT_DIR / "part1_highboost_k2.png", sharpened[2.0])
    save_grayscale(OUTPUT_DIR / "part1_highboost_k4_5.png", sharpened[4.5])
    # Make a labeled comparison figure for the report.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (image, "Original grayscale image"),
        (np.clip(blurred, 0, 255).astype(np.uint8),
         f"Gaussian blur (kernel={BLUR_KERNEL_SIZE}, sigma={BLUR_SIGMA})"),
        (mask_display, f"Unsharp mask (shifted by 128, gain = {MASK_GAIN})"),
        (sharpened[1.0], "Unsharp masking (k = 1)"),
        (sharpened[2.0], "High-boost filtering (k = 2)"),
        (sharpened[4.5], "High-boost filtering (k = 4.5)"),
    ]
    for ax, (panel_image, title) in zip(axes.flat, panels):
        ax.imshow(panel_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Part 1: Unsharp Masking and High-Boost Filtering", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part1_sharpening_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved Part 1 comparison and intermediate images.")
def part2_canny_edge_detection(image: np.ndarray) -> None:
    """
    Compare Canny edge maps using different Gaussian smoothing scales and
    hysteresis threshold pairs. OpenCV's Canny is used for the multi-stage
    edge detection; Gaussian blur is applied beforehand to control sigma.
    """
    print("\nPART 2: Canny edge detector")
    results = []
    for config in CANNY_CONFIGS:
        sigma = config["sigma"]
        low = config["low"]
        high = config["high"]
        # Build an odd kernel size large enough for the chosen sigma.
        # A common practical rule is approximately 6*sigma, rounded to odd.
        kernel_size = max(3, int(round(6 * sigma)) | 1)
        # Explicit Gaussian smoothing lets us compare small and large scales.
        smoothed = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), sigmaX=sigma
        )
        # Canny then performs gradient estimation, non-maximum suppression,
        # double thresholding, and hysteresis edge tracking internally.
        edges = cv2.Canny(smoothed, threshold1=low, threshold2=high)
        filename = (
            f"part2_canny_sigma{sigma}_low{low}_high{high}.png"
        )
        save_grayscale(OUTPUT_DIR / filename, edges)
        results.append((config, edges))
        print(f"Saved {filename}")
    # --- NMS and hysteresis demonstration (sigma = 0.8) ---
    smoothed = cv2.GaussianBlur(image, (5, 5), 0.8)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    no_nms = (mag > 150).astype(np.uint8) * 255   # threshold only, no NMS
    single = cv2.Canny(smoothed, 150, 150)        # one threshold, no hysteresis linking
    double = cv2.Canny(smoothed, 50, 150)         # full Canny
    demo = [
        (no_nms, "Gradient magnitude > 150 (no NMS)"),
        (single, "Canny, T_low = T_high = 150 (no hysteresis)"),
        (double, "Canny, [50, 150] (NMS + hysteresis)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, (img, title) in zip(axes, demo):
        ax.imshow(255 - img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Role of non-maximum suppression and hysteresis", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part2_nms_hysteresis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Display only the six Canny edge maps in a 2-by-3 grid.
    # The original grayscale image is already shown in the Part 1 figure.
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    for ax, (config, edges) in zip(axes.flat, results):
        ax.imshow(255 - edges, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"σ={config['sigma']}, "
            f"Tlow={config['low']}, Thigh={config['high']}",
            fontsize=9
        )
        ax.axis("off")
    fig.suptitle("Part 2: Canny Edge Detection Parameter Comparison",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part2_canny_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved Part 2 comparison grid.")
def main() -> None:
    """Create output folder, load the image, and run both assignment parts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_grayscale_image(INPUT_PATH)
    print(f"Loaded image: {INPUT_PATH}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
    part1_unsharp_and_highboost(image)
    part2_canny_edge_detection(image)
    print("\nFinished. Check the 'output' folder for figures and images.")
    print("Use the comparison PNGs as figures in your PDF report.")
if __name__ == "__main__":
    main()
