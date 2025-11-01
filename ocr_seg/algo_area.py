import cv2
import numpy as np

'''def show(name, img, width=800, height=600):
    """Show image in fixed-size window."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, img)'''

# Load & Preprocess

img = cv2.imread('images/10388.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE -> better contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)
cv2.imshow("0. CLAHE", gray_eq)
cv2.waitKey(0)

# Remove shadows
kernel_init = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
print(f"Initial kernel size: {kernel_init.shape[0]}x{kernel_init.shape[1]}")
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_init)
diff = cv2.absdiff(gray_eq, bg)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("1. Shadow Removed", norm_img)
cv2.waitKey(0)

# Blur + Otsu
blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
_, binarized_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("2. Binarized", binarized_img)
cv2.waitKey(0)

# Connected Components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binarized_img, connectivity=8)

# Random color map
label_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for i in range(1, num_labels):
    mask = labels == i
    color = np.random.randint(0, 255, size=3).tolist()
    label_img[mask] = color
cv2.imshow("3. Connected Components", label_img)
cv2.waitKey(0)

# Filter small blobs
min_area = 1000
refined = binarized_img.copy()
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        refined[labels == i] = 0
cv2.imshow("4. Filtered Large Components", refined)
cv2.waitKey(0)


# Auto Kernel Selection (Area + Height based)

areas = stats[1:, cv2.CC_STAT_AREA]
heights = stats[1:, cv2.CC_STAT_HEIGHT]
avg_area = np.mean(areas) if len(areas) > 0 else 500
avg_height = np.mean(heights) if len(heights) > 0 else 20

# Base kernel from both
side = int(((np.sqrt(avg_area) * 0.6) + (avg_height * 0.4)) // 16)
side = max(3, side | 1)
kernel_auto = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
print(f"\nAuto kernel size: {side}x{side} (from avg area={avg_area:.1f}, avg height={avg_height:.1f})")

# Print per-blob kernel sizes
print("\nKernel size for each blob:")
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    if area < min_area:
        continue
    # combine both
    side_i = int(((np.sqrt(area) * 0.6) + (height * 0.4)) // 16)
    side_i = max(3, side_i | 1)
    print(f"Blob {i}: area={area}, height={height}, kernel={side_i}x{side_i}")

# Morph closing using auto-tuned kernel
dilated = cv2.dilate(refined, kernel_auto, iterations=1)
closed_auto = cv2.erode(dilated, kernel_auto, iterations=1)
cv2.imshow("5. Auto-Tuned Closing", closed_auto)
cv2.waitKey(0)
cv2.destroyAllWindows()
