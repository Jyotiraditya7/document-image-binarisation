import cv2

myImage = cv2.imread('images/shadow_side.jpg')  # Replace with your file path
cv2.imshow('Original Image', myImage)
cv2.waitKey(0)

# Convert to Grayscale
grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", grayImg)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(grayImg, (5, 5), 0)

# Apply general threshold using otsu's method
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,10))
dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow("Dilated", dilation)
cv2.waitKey(0)

# Find contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im2 = myImage.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 50:  # filter out small noise
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 0), 5)

cv2.imshow("Segmented", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
