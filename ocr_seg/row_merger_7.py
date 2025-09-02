import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_image(img_path):
    """Your preprocessing pipeline - works well"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    
    # Shadow removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
    diff = cv2.absdiff(gray_eq, bg)
    norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Binarization
    _, binary = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img, binary

def detect_lines_dbscan(binary_img):
    """Your DBSCAN line detection - keep this as is"""
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes
    boxes = []
    centers_y = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 3000:  # ignore small noise
            boxes.append([x, y, x + w, y + h])
            centers_y.append([y + h // 2])  # Use Y center only for vertical clustering
    
    if not centers_y:
        return []
    
    # Apply DBSCAN to cluster boxes into rows
    db = DBSCAN(eps=40, min_samples=1).fit(centers_y)
    labels = db.labels_
    
    # Group boxes by cluster
    clustered_boxes = {}
    for label, box in zip(labels, boxes):
        clustered_boxes.setdefault(label, []).append(box)
    
    # Get line boundaries for each cluster
    line_regions = []
    for box_group in clustered_boxes.values():
        y1 = min([b[1] for b in box_group])
        y2 = max([b[3] for b in box_group])
        line_regions.append((y1, y2))
    
    # Sort lines top to bottom
    line_regions = sorted(line_regions, key=lambda x: x[0])
    return line_regions

def segment_words_in_line(binary_img, line_bounds):
    """Segment words within a detected line using vertical projection"""
    y1, y2 = line_bounds
    
    # Add small padding to capture diacritics
    padding = 5
    y1_pad = max(0, y1 - padding)
    y2_pad = min(binary_img.shape[0], y2 + padding)
    
    line_img = binary_img[y1_pad:y2_pad, :]
    h_line, w_line = line_img.shape
    
    if h_line == 0 or w_line == 0:
        return []
    
    # Vertical projection (count white pixels per column)
    vertical_proj = np.sum(line_img > 0, axis=0)
    
    # Find word boundaries using valley detection
    # Smooth projection to avoid noise
    if len(vertical_proj) > 5:
        vertical_proj = cv2.GaussianBlur(vertical_proj.astype(np.float32).reshape(1, -1), (5, 1), 0).flatten()
    
    # Find valleys (gaps between words)
    max_val = np.max(vertical_proj)
    if max_val == 0:
        return []
    
    # Valley threshold - areas with very few pixels
    valley_threshold = max_val * 0.1  # 10% of maximum
    
    # Find word regions
    word_regions = []
    in_word = False
    start_x = 0
    min_word_width = 15  # Minimum width for a word
    
    for i, val in enumerate(vertical_proj):
        if val > valley_threshold and not in_word:
            start_x = i
            in_word = True
        elif val <= valley_threshold and in_word:
            # End of word found
            if i - start_x >= min_word_width:
                word_regions.append((start_x, y1_pad, i, y2_pad))
            in_word = False
    
    # Handle last word if it extends to the end
    if in_word and w_line - start_x >= min_word_width:
        word_regions.append((start_x, y1_pad, w_line, y2_pad))
    
    return word_regions

def refine_word_boxes(binary_img, word_regions):
    """Refine word boxes by finding actual content boundaries"""
    refined_boxes = []
    
    for x1, y1, x2, y2 in word_regions:
        # Extract word region
        word_roi = binary_img[y1:y2, x1:x2]
        
        if word_roi.size == 0:
            continue
        
        # Find the actual content boundaries within this region
        # Horizontal bounds
        col_sum = np.sum(word_roi > 0, axis=0)
        nonzero_cols = np.where(col_sum > 0)[0]
        if len(nonzero_cols) == 0:
            continue
        
        # Vertical bounds  
        row_sum = np.sum(word_roi > 0, axis=1)
        nonzero_rows = np.where(row_sum > 0)[0]
        if len(nonzero_rows) == 0:
            continue
        
        # Get tight bounding box
        tight_x1 = x1 + nonzero_cols[0]
        tight_x2 = x1 + nonzero_cols[-1] + 1
        tight_y1 = y1 + nonzero_rows[0]
        tight_y2 = y1 + nonzero_rows[-1] + 1
        
        # Add small padding
        padding = 2
        final_x1 = max(0, tight_x1 - padding)
        final_y1 = max(0, tight_y1 - padding)
        final_x2 = min(binary_img.shape[1], tight_x2 + padding)
        final_y2 = min(binary_img.shape[0], tight_y2 + padding)
        
        # Only keep if reasonable size
        if (final_x2 - final_x1) * (final_y2 - final_y1) > 200:
            refined_boxes.append((final_x1, final_y1, final_x2, final_y2))
    
    return refined_boxes

def split_wide_words(binary_img, word_boxes, max_width_ratio=6):
    """Split words that are too wide (likely merged words)"""
    final_boxes = []
    
    for x1, y1, x2, y2 in word_boxes:
        width = x2 - x1
        height = y2 - y1
        
        # If word is too wide relative to height, try to split it
        if width > height * max_width_ratio and width > 100:
            # Extract word region
            word_roi = binary_img[y1:y2, x1:x2]
            
            # Vertical projection for splitting
            vertical_proj = np.sum(word_roi > 0, axis=0)
            
            # Find valleys for splitting
            max_val = np.max(vertical_proj)
            valley_threshold = max_val * 0.05  # Very low threshold for splitting
            
            # Find split points
            splits = []
            min_gap = 8  # Minimum gap width to consider as word boundary
            
            in_valley = False
            valley_start = 0
            
            for i, val in enumerate(vertical_proj):
                if val <= valley_threshold and not in_valley:
                    valley_start = i
                    in_valley = True
                elif val > valley_threshold and in_valley:
                    valley_width = i - valley_start
                    if valley_width >= min_gap:
                        split_point = valley_start + valley_width // 2
                        splits.append(split_point)
                    in_valley = False
            
            # Create sub-words
            if splits:
                prev_x = 0
                for split_x in splits:
                    if split_x - prev_x > 20:  # Minimum sub-word width
                        final_boxes.append((x1 + prev_x, y1, x1 + split_x, y2))
                    prev_x = split_x
                
                # Add last segment
                if width - prev_x > 20:
                    final_boxes.append((x1 + prev_x, y1, x2, y2))
            else:
                # No good split found, keep original
                final_boxes.append((x1, y1, x2, y2))
        else:
            # Word is reasonable width, keep as is
            final_boxes.append((x1, y1, x2, y2))
    
    return final_boxes

def main():
    # Load and preprocess
    img, binary = preprocess_image('images/slant.jpg')
    
    print(f"Image shape: {img.shape}")
    cv2.imshow("1. Binary Image", binary)
    cv2.waitKey(0)
    
    # Detect lines using your DBSCAN method
    line_regions = detect_lines_dbscan(binary)
    print(f"Found {len(line_regions)} text lines")
    
    if len(line_regions) == 0:
        print("No lines detected!")
        return []
    
    # Visualize detected lines
    line_vis = img.copy()
    for i, (y1, y2) in enumerate(line_regions):
        cv2.rectangle(line_vis, (0, y1), (img.shape[1]-1, y2), (0, 255, 0), 3)
        cv2.putText(line_vis, f'Line {i+1}', (10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("2. Detected Lines", line_vis)
    cv2.waitKey(0)
    
    # Segment words in each line
    all_word_boxes = []
    for i, line_region in enumerate(line_regions):
        print(f"Processing line {i+1}: y-range {line_region}")
        
        # Get word regions using vertical projection
        word_regions = segment_words_in_line(binary, line_region)
        print(f"  Initial words in line {i+1}: {len(word_regions)}")
        
        # Refine boxes to tight fit
        refined_words = refine_word_boxes(binary, word_regions)
        print(f"  Refined words in line {i+1}: {len(refined_words)}")
        
        all_word_boxes.extend(refined_words)
    
    print(f"Total initial word boxes: {len(all_word_boxes)}")
    
    # Split any overly wide words
    final_boxes = split_wide_words(binary, all_word_boxes)
    print(f"Final word count after splitting: {len(final_boxes)}")
    
    # Draw final result
    result = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Add word numbers
        cv2.putText(result, str(i+1), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imshow("3. Final Word Segmentation", result)
    cv2.waitKey(0)
    
    # Also show on binary for clarity
    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in final_boxes:
        cv2.rectangle(binary_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    cv2.imshow("4. Words on Binary", binary_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_boxes

if __name__ == "__main__":
    print("Method 1: Line detection + word segmentation")
    boxes1 = main()
    
    print(f"\nMethod 1 result: {len(boxes1)} words detected")