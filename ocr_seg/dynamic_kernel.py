import cv2
import numpy as np

def oddize(x):
    x = int(round(x))
    if x <= 1:
        return 3
    return x if x % 2 == 1 else x + 1

def compute_band_kernel_sizes(stats, centroids, img_h, num_bands=30,
                              area_thresh=50, scale_factor=0.5,
                              min_k=3, max_k=101):
    """
    stats, centroids from connectedComponentsWithStats.
    Return list of kernel sizes (one per band).
    scale_factor controls kernel relative to component height.
    """
    # collect components (skip background index 0)
    comp_heights = []
    comp_centroids_y = []
    n = stats.shape[0]
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_thresh:
            continue
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy = centroids[i][1]
        comp_heights.append(h)
        comp_centroids_y.append(cy)

    # if no components pass filter, fallback to global median of all heights
    if len(comp_heights) == 0:
        return [oddize(min(max(min_k, 5), max_k))] * num_bands

    comp_heights = np.array(comp_heights)
    comp_centroids_y = np.array(comp_centroids_y)

    band_kernel_sizes = []
    band_height = img_h / num_bands
    global_med = np.median(comp_heights)

    for b in range(num_bands):
        y0 = b * band_height
        y1 = (b + 1) * band_height
        mask = (comp_centroids_y >= y0) & (comp_centroids_y < y1)
        if mask.sum() >= 1:
            med_h = np.median(comp_heights[mask])
        else:
            # if no components in this band, interpolate with neighbors by using global median
            med_h = global_med

        # kernel size proportional to median component height
        k = int(round(med_h * scale_factor))
        k = max(min_k, min(k, max_k))
        k = oddize(k)
        band_kernel_sizes.append(k)

    return band_kernel_sizes

def dynamic_closing_by_bands(gray_img, band_kernel_sizes, overlap=0.2):
    """
    Apply closing per horizontal band, using kernel sizes from band_kernel_sizes.
    overlap: fraction of band height to overlap with neighbor (0..0.5)
    Returns the closed image (same size).
    """
    h, w = gray_img.shape[:2]
    num_bands = len(band_kernel_sizes)
    band_h = h / num_bands
    out = np.zeros_like(gray_img)

    # we will keep a weight map to blend overlapping regions (max-blend)
    for b in range(num_bands):
        ksize = band_kernel_sizes[b]
        # compute integer band region including overlap
        start = int(round(max(0, (b - overlap) * band_h)))
        end = int(round(min(h, (b + 1 + overlap) * band_h)))
        band_slice = gray_img[start:end]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed = cv2.morphologyEx(band_slice, cv2.MORPH_CLOSE, kernel)

        # place closed into out by max to avoid seams
        existing = out[start:end]
        blended = np.maximum(existing, closed)
        out[start:end] = blended

    return out

def visualize_bands(img, band_kernel_sizes, overlap=0.2):
    """
    Draw horizontal bands on the image to visualize band ranges and kernel sizes.
    img: grayscale or color image (will be converted to BGR if grayscale)
    band_kernel_sizes: list of kernel sizes per band
    """
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h = vis.shape[0]
    num_bands = len(band_kernel_sizes)
    band_h = h / num_bands

    for b, k in enumerate(band_kernel_sizes):
        start = int(round(max(0, (b - overlap) * band_h)))
        end = int(round(min(h, (b + 1 + overlap) * band_h)))
        # draw band region in semi-transparent way
        color = (0, 0, 255)  # red for visualization
        cv2.rectangle(vis, (0, start), (vis.shape[1]-1, end), color, 1)
        cv2.putText(vis, f"k={k}", (5, start+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Bands Visualization", vis)
    cv2.waitKey(0)


# ------------------ Main pipeline (based on your code) ------------------
img = cv2.imread('images/shadow_side.jpg')
if img is None:
    raise SystemExit("Failed to read images/slant.jpg — check path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # convert to grayscale
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)
cv2.imshow("0. CLAHE", gray_eq)
cv2.waitKey(0)

# initial coarse background-removal using a relatively large fixed kernel to get diff
kernel_init = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
bg_init = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_init)
diff = cv2.absdiff(gray_eq, bg_init)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("1. Shadow Removed (norm)", norm_img)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(norm_img, (5,5), 0)
_, binarized_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("2. Binarized (OTSU)", binarized_img)
cv2.waitKey(0)

# small morphological open to clean speckle (optional)
small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binarized_img = cv2.morphologyEx(binarized_img, cv2.MORPH_OPEN, small_k)

# compute connected components to estimate local text height
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_img, connectivity=8)

# choose parameters for kernel estimation
num_bands = 40                 # more bands => finer spatial adaptivity
area_thresh = 30               # ignore tiny specks
scale_factor = 0.6             # kernel ≈ 0.6 * median component height (tune this)
min_k = 3
max_k = 101

band_k = compute_band_kernel_sizes(stats, centroids, binarized_img.shape[0],
                                  num_bands=num_bands,
                                  area_thresh=area_thresh,
                                  scale_factor=scale_factor,
                                  min_k=min_k,
                                  max_k=max_k)

for i in range(len(band_k)):
    print(f"Band {i}: kernel size = {band_k[i]}")

visualize_bands(gray_eq, band_k, overlap=0.35)
 
# Apply dynamic closing by bands
closed_dynamic = dynamic_closing_by_bands(binarized_img, band_k, overlap=0.35)
cv2.imshow("3. Dynamic Closing (by bands)", closed_dynamic)
cv2.waitKey(0)

# Post-process and keep large components as you did
num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(closed_dynamic, connectivity=8)

min_area = 1000
refined = closed_dynamic.copy()
# for i in range(1, num_labels2):
#     if stats2[i, cv2.CC_STAT_AREA] < min_area:
#         refined[labels2 == i] = 0

# cv2.imshow("4. Filtered Large Components", refined)
# cv2.waitKey(0)

output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)
word_id = 0
for i in range(1, num_labels2):
    x, y, w, h, area = stats2[i]
    if area >= min_area:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        word_crop = refined[y:y+h, x:x+w]
        word_id += 1

cv2.imshow("Word Boxes (Connected Components)", output)
cv2.waitKey(0)

finalimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
word_id=0
for i in range(1,num_labels2):
    x,y,w,h,area = stats2[i]
    if area >= min_area:
        cv2.rectangle(finalimg,(x,y),(x+w,y+h),(0,255,0),2)
        word_crop = refined[y:y+h, x:x+w]
        word_id+=1

cv2.imshow("Word Boxes (Connected Components)", finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
