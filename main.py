import cv2
import pytesseract

# Load image
img = cv2.imread("sample_images/car.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 2 < w / h < 6 and w > 100:
        plate = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(plate, config="--psm 8")
        print("Detected Plate:", text.strip())
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("License Plate Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
