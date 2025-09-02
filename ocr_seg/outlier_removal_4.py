import cv2
import numpy as np

img = cv2.imread('images/shadow_bottom.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # convert to grayscale

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # makes text strokes more visible
gray_eq = clahe.apply(gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))    # Morphological closing (dilation + erosion)
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
diff = cv2.absdiff(gray_eq, bg)                                     # removes shadows, leaving mostly text strokes.

norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("1. Shadow Removed", norm_img)
cv2.waitKey(0)

_, rough_thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Global Otsu thresholding

contours, _ = cv2.findContours(rough_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # gives candidate text blobs


formatted = np.zeros_like(gray)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area < 2000:
        continue

    roi = norm_img[y:y+h, x:x+w]   # region of interest

    roi_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    dist = cv2.distanceTransform(roi_bin, cv2.DIST_L2, 5)           # Use distance transform to measure thickness

    mean_stroke = int(np.mean(dist[dist > 0]) * 2) | 1              # mean stroke width to adapt kernel

    k = max(15, mean_stroke)
    k = k if k % 2 == 1 else k + 1

    local_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,    # Apply adaptive threshold with dynamic block size k
                                      cv2.THRESH_BINARY, k, 10)

    formatted[y:y+h, x:x+w] = local_bin                                 # replace ROI

cv2.imshow("2. Adaptive Local Binarization", formatted)
cv2.waitKey(0)

contours, _ = cv2.findContours(formatted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours again from the refined binarized image.

adaptive_closed = formatted.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)           # Perform morphological closing
    area = w * h
    if area > 3000:
        k_w = max(3, int(w / 10)) | 1
        k_h = max(3, int(h / 4)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        roi = adaptive_closed[y:y+h, x:x+w]
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        adaptive_closed[y:y+h, x:x+w] = closed

cv2.imshow("3. Adaptive Morph Closing", adaptive_closed)
cv2.waitKey(0)

contours, _ = cv2.findContours(adaptive_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours again

merged_boxes = []
line_tolerance = 30                 # vertical tolerance
gap_tolerance = 5                   # horizontal tolerance

for cnt in contours:
    x1, y1, w1, h1 = cv2.boundingRect(cnt)
    if w1 * h1 < 2000:                      # skip very small (noise)
        continue

    box1 = [x1, y1, x1 + w1, y1 + h1]
    cy1 = y1 + h1 // 2
    merged = False

    for j, box2 in enumerate(merged_boxes):
        xa1, ya1, xa2, ya2 = box2
        cy2 = ya1 + (ya2 - ya1) // 2

        if abs(cy1 - cy2) < line_tolerance:

            if not (box1[2] < xa1 - gap_tolerance or box1[0] > xa2 + gap_tolerance):
                new_box = [
                    min(xa1, box1[0]),
                    min(ya1, box1[1]),
                    max(xa2, box1[2]),
                    max(ya2, box1[3])
                ]
                merged_boxes[j] = new_box
                merged = True                           # merge if very close horizontally
                break

    if not merged:
        merged_boxes.append(box1)


pre_img = img.copy()
for x1, y1, x2, y2 in merged_boxes:
    cv2.rectangle(pre_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv2.imshow("4. Merged Contours (before outlier removal)", pre_img)      # Show before outlier removal
cv2.waitKey(0)




# ---------------------------------------------------------------------------------





# OUTLIER REMOVAL (SPATIAL CLUSTER)

cleaned_boxes = merged_boxes[:]
n = len(merged_boxes)
if n > 0:
    centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in merged_boxes])

    heights = np.array([(y2 - y1) for (x1, y1, x2, y2) in merged_boxes])        # Compute centers and heights of boxes

    R = int(max(20, np.median(heights) * 4.0))
    R2 = R * R                                               # radius for neighborhood

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if dx * dx + dy * dy <= R2:
                adj[i].append(j)                            # Boxes within radius R are considered connected
                adj[j].append(i)

    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                u = stack.pop()
                for v in adj[u]:                            # Find connected components of boxes using DFS
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                        comp.append(v)
            components.append(comp)

    if components:
        largest = max(components, key=len)
        keep = set(largest)
        cleaned_boxes = [merged_boxes[i] for i in range(n) if i in keep]





output = img.copy()
for x1, y1, x2, y2 in cleaned_boxes:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), 10)

cv2.imshow("5. Final Merged Contours (Outliers Removed)", output)       # Draw Final Merged Boxes

cv2.waitKey(0)
cv2.destroyAllWindows()
