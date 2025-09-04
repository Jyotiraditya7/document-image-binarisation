import cv2
import numpy as np

img = cv2.imread('images/slant.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)
cv2.imshow("1. CLAHE", gray_eq)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
diff = cv2.absdiff(gray_eq, bg)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("2. Shadow Removed", norm_img)
cv2.waitKey(0)

_, light_thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("3. Threshold", light_thresh)
cv2.waitKey(0)


h, w = light_thresh.shape
horizontal_sum = np.sum(light_thresh > 0, axis=1)
min_pixels_per_row = max(10, int(0.005 * w))
row_presence = (horizontal_sum >= min_pixels_per_row).astype(np.uint8)

rows = []
in_run = False
start = 0
for i, v in enumerate(row_presence):
    if v and not in_run:
        start = i
        in_run = True
    elif not v and in_run:
        rows.append((start, i - 1))
        in_run = False
if in_run:
    rows.append((start, len(row_presence) - 1))


row_vis = cv2.cvtColor(light_thresh, cv2.COLOR_GRAY2BGR)
for (r1, r2) in rows:
    cv2.rectangle(row_vis, (0, r1), (w - 1, r2), (0, 255, 0), 1)
cv2.imshow("4. Row Bands", row_vis)
cv2.waitKey(0)


def merge_boxes_horiz(boxes, gap=10, overlap_thresh=0.25):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0].copy()]
    for x1, y1, x2, y2 in boxes[1:]:
        xa1, ya1, xa2, ya2 = merged[-1]
        inter = max(0, min(xa2, x2) - max(xa1, x1))
        minw = min(xa2 - xa1, x2 - x1)
        if (minw > 0 and (inter / float(minw) > overlap_thresh)) or (x1 - xa2) <= gap:
            merged[-1] = [min(xa1, x1), min(ya1, y1), max(xa2, x2), max(ya2, y2)]
        else:
            merged.append([x1, y1, x2, y2])
    return merged

def split_box_by_vproj(bin_img_full, box, min_gap_pixels=8, valley_frac=0.08):
    x1, y1, x2, y2 = box
    roi = bin_img_full[y1:y2, x1:x2]
    if roi.size == 0:
        return [box]

    col_sum = np.sum(roi > 0, axis=0).astype(np.float32)
    if col_sum.max() <= 0:
        return [box]
    norm = col_sum / (col_sum.max() + 1e-8)
    valley_mask = (norm <= valley_frac).astype(np.uint8)

    splits = []
    in_valley = False
    vstart = 0
    for i, v in enumerate(valley_mask):
        if v and not in_valley:
            vstart = i
            in_valley = True
        elif not v and in_valley:
            vend = i - 1
            if vend - vstart + 1 >= min_gap_pixels:
                splits.append((vstart, vend))
            in_valley = False
    if in_valley:
        vend = len(valley_mask) - 1
        if vend - vstart + 1 >= min_gap_pixels:
            splits.append((vstart, vend))

    if not splits:
        return [box]

    # choose split x positions at middle of valley runs
    split_xs = [ (s+e)//2 for (s,e) in splits ]
    # produce subboxes
    subboxes = []
    prev = 0
    for sx in split_xs:
        left_x = x1 + prev
        right_x = x1 + sx
        if right_x - left_x > 5:
            subboxes.append([left_x, y1, right_x, y2])
        prev = sx
    # last segment
    left_x = x1 + prev
    right_x = x2
    if right_x - left_x > 5:
        subboxes.append([left_x, y1, right_x, y2])

    # if splitting produced nothing meaningful, return original
    if len(subboxes) == 0:
        return [box]
    return subboxes


final_boxes = []
for (r1, r2) in rows:
    # crop one row band (add small padding vertically to capture diacritics)
    pad = 3
    top = max(0, r1 - pad)
    bottom = min(h - 1, r2 + pad)
    line_img = light_thresh[top:bottom+1, :].copy()

    # small opening to remove specks and thin vertical bridges
    kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    line_img = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kern_open, iterations=1)

    # horizontal closing to connect characters into words but only inside row (1-pixel tall kernel to avoid vertical merge)
    # kernel width tuned relative to image width (adaptive)
    adaptive_kw = max(15, int(w / 30))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (adaptive_kw, 1))
    line_closed = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, kernel_h, iterations=1)

    # find contours inside this row band
    contours_row, _ = cv2.findContours(line_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_boxes = []
    for cnt in contours_row:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 500:   # small threshold inside row
            continue
        # convert to full image coords
        row_boxes.append([x, top + y, x + bw, top + y + bh])

    # merge boxes horizontally within this row
    merged_row = merge_boxes_horiz(row_boxes, gap=12, overlap_thresh=0.2)

    # for each merged box, if it's suspiciously wide (likely contains multiple words), try splitting via vertical projection on the original (not row-closed) binary
    for mb in merged_row:
        x1, y1, x2, y2 = mb
        bw = x2 - x1
        bh = y2 - y1
        if bw > max(200, 5 * bh):   # heuristics: very wide compared to height => attempt split
            subs = split_box_by_vproj(light_thresh, mb, min_gap_pixels=max(6, int(bw*0.01)), valley_frac=0.07)
            # further refine: discard very small subs
            for s in subs:
                sx1, sy1, sx2, sy2 = s
                if (sx2-sx1)*(sy2-sy1) >= 300:
                    final_boxes.append(s)
        else:
            final_boxes.append(mb)

# preview merged boxes before outlier removal
pre_img = img.copy()
for x1, y1, x2, y2 in final_boxes:
    cv2.rectangle(pre_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow("Merged Boxes per Row (before outlier removal)", pre_img)
cv2.waitKey(0)

# ================== OUTLIER REMOVAL (SPATIAL CLUSTER) ==================
cleaned_boxes = final_boxes[:]
n = len(final_boxes)
if n > 0:
    centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in final_boxes])
    heights = np.array([(y2 - y1) for (x1, y1, x2, y2) in final_boxes])

    R = int(max(20, np.median(heights) * 4.0))
    R2 = R * R

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if dx * dx + dy * dy <= R2:
                adj[i].append(j)
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
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                        comp.append(v)
            components.append(comp)

    if components:
        largest = max(components, key=len)
        keep = set(largest)
        cleaned_boxes = [final_boxes[i] for i in range(n) if i in keep]

# ================== FINAL DRAW ==================
output = img.copy()
for x1, y1, x2, y2 in cleaned_boxes:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), 8)

cv2.imshow("Final Merged Contours (Outliers Removed)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
