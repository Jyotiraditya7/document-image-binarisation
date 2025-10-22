import cv2
import numpy as np

def oddize(x):
    x = int(round(x))
    if x <= 1:
        return 3
    return x if x % 2 == 1 else x + 1

def smooth_list(arr, kernel=3):
    if kernel <= 1:
        return np.array(arr, dtype=float)
    k = np.ones(kernel) / kernel
    pad = kernel // 2
    a = np.pad(arr, pad, mode='edge')
    sm = np.convolve(a, k, mode='valid')
    return sm

def compute_band_kernel_sizes(stats, centroids, img_h,
                              num_bands=40,
                              area_thresh=30,
                              scale_factor=0.55,
                              min_k=3, max_k=101,
                              density_shrink_coeff=0.6):
    """
    Estimate kernel sizes per horizontal band from connected component stats.
    - stats, centroids: from connectedComponentsWithStats
    - img_h: image height (pixels)
    Returns: list of odd integer kernel sizes, length = num_bands
    """
    comp_heights = []
    comp_centroids_y = []
    n = stats.shape[0]
    for i in range(1, n):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_thresh:
            continue
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy = centroids[i][1]
        comp_heights.append(h)
        comp_centroids_y.append(cy)

    # fallback if no components
    if len(comp_heights) == 0:
        return [oddize(min(max(min_k, 5), max_k))] * num_bands

    comp_heights = np.array(comp_heights)
    comp_centroids_y = np.array(comp_centroids_y)

    band_kernel_sizes = []
    band_h = img_h / num_bands
    global_med = np.median(comp_heights)

    # precompute density normalization factor
    # roughly components per band (not normalized by area); we'll use it to shrink kernels in dense areas
    for b in range(num_bands):
        y0 = b * band_h
        y1 = (b + 1) * band_h
        mask = (comp_centroids_y >= y0) & (comp_centroids_y < y1)
        n_in_band = int(mask.sum())
        if n_in_band >= 1:
            med_h = np.median(comp_heights[mask])
            # density = components per band height (higher means more characters close together)
            density = n_in_band / band_h
        else:
            med_h = global_med
            density = 0.0

        # base kernel from median component height
        k = float(med_h) * scale_factor

        # shrink kernel if density is high to avoid merging adjacent words/characters
        # density_shrink_coeff controls how aggressively to shrink; 0=no shrink, larger ~ more shrink
        shrink_scale = 1.0 / (1.0 + density_shrink_coeff * density * 50.0)
        k = k * shrink_scale

        # clamp
        k = max(min_k, min(k, max_k))
        k = oddize(k)
        band_kernel_sizes.append(k)

    # smooth kernel sizes so that neighboring bands don't jump abruptly
    smoothed = smooth_list(band_kernel_sizes, kernel=5)
    smoothed = np.clip(smoothed, min_k, max_k)
    smoothed = [oddize(x) for x in smoothed]
    return smoothed

# ------------------------ Dynamic closing (per-band) -----------------------
def dynamic_closing_by_bands(gray_img, band_kernel_sizes, overlap=0.30):
    """
    Apply morphological closing per horizontal band and max-blend results.
    - gray_img: binary (0/255) or grayscale image
    - band_kernel_sizes: list[int] kernel sizes (height), length = num_bands
    - overlap: fraction of band height to overlap with neighbors (0..0.5)
    Returns closed image (same dtype/shape as gray_img)
    """
    h, w = gray_img.shape[:2]
    num_bands = len(band_kernel_sizes)
    band_h = h / num_bands
    out = np.zeros_like(gray_img)

    for b in range(num_bands):
        ksize = int(band_kernel_sizes[b])

        # region bounds including overlap
        start = int(round(max(0, (b - overlap) * band_h)))
        end = int(round(min(h, (b + 1 + overlap) * band_h)))
        if end <= start:
            continue

        band_slice = gray_img[start:end]

        # Anisotropic elliptical kernel: narrower horizontally to avoid merging words.
        # width ~ max(3, ksize // 2), height ~ ksize
        k_w = max(3, ksize // 2)
        k_h = max(3, ksize)
        # ensure odd sizes
        k_w = oddize(k_w)
        k_h = oddize(k_h)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
        closed = cv2.morphologyEx(band_slice, cv2.MORPH_CLOSE, kernel)

        # max-blend to avoid seam artifacts and preserve strongest white pixels
        existing = out[start:end]
        out[start:end] = np.maximum(existing, closed)

    return out

# ------------------------------ Main pipeline ------------------------------
def main():
    # ----- Parameters you can tune -----
    image_path = "images/shadow_side.jpg"   # change if needed
    num_bands = 40
    area_thresh = 30         # ignore tiny specks when estimating scale
    scale_factor = 0.55      # kernel ≈ scale_factor * median component height
    density_shrink_coeff = 0.6
    min_k = 3
    max_k = 121
    overlap = 0.30
    min_area_keep = 1000     # keep only components larger than this after closing
    # -----------------------------------

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Failed to read '{image_path}' — check file path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) CLAHE -> helps text strokes
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    cv2.imshow("0. CLAHE", gray_eq); cv2.waitKey(0)

    # 2) coarse background (large fixed closing) to remove gradients/shadows
    kernel_init = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bg_init = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_init)
    diff = cv2.absdiff(gray_eq, bg_init)
    norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("1. Shadow Removed (norm)", norm_img); cv2.waitKey(0)

    # 3) smooth + Otsu binarize
    blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("2. Binarized (OTSU)", binarized); cv2.waitKey(0)

    # small open to remove speckle
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, small_k)

    # 4) get CC stats to estimate local heights
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)

    # compute per-band kernel sizes (custom logic)
    band_k = compute_band_kernel_sizes(stats, centroids, binarized.shape[0],
                                      num_bands=num_bands,
                                      area_thresh=area_thresh,
                                      scale_factor=scale_factor,
                                      min_k=min_k, max_k=max_k,
                                      density_shrink_coeff=density_shrink_coeff)

    print("Sample band kernels:", band_k[:8], "...", band_k[-6:])

    # 5) dynamic closing per-band
    closed_dynamic = dynamic_closing_by_bands(binarized, band_k, overlap=overlap)
    cv2.imshow("3. Dynamic Closing (by bands)", closed_dynamic); cv2.waitKey(0)

    # 6) connected components after closing -> filter small components
    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(closed_dynamic, connectivity=8)
    refined = closed_dynamic.copy()
    # for i in range(1, num_labels2):
    #     if stats2[i, cv2.CC_STAT_AREA] < min_area_keep:
    #         refined[labels2 == i] = 0

    cv2.imshow("4. Filtered Large Components", refined); cv2.waitKey(0)

    # 7) draw bounding boxes for remaining components
    output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)
    word_id = 0
    for i in range(1, num_labels2):
        x, y, w, h, area = stats2[i]
        if area >= min_area_keep:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            word_id += 1

    print("Word/region count (kept):", word_id)
    cv2.imshow("Word Boxes (Connected Components)", output); cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
