import cv2
import numpy as np

# --- Settings ---
KNOWN_WIDTH_CM = 21.0  # Width of your reference object (e.g., A4 paper)

def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    # find the largest contour and use it as reference
    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)  # returns (center(x, y), (width, height), angle)

def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

def detect_shapes_and_dimensions(frame, pixels_per_cm):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            width_cm = round(w / pixels_per_cm, 2)
            height_cm = round(h / pixels_per_cm, 2)

            shape = "Unknown"
            corners = len(approx)
            if corners == 3:
                shape = "Triangle"
            elif corners == 4:
                shape = "Rectangle"
            elif corners == 4:
                shape = "Square"
            elif corners == 5:
                shape = "Pentagon"
            elif corners == 6:
                shape = "Hexagon"
            elif corners == 7:
                shape = "Heptagone"
            elif corners == 8:
                shape = "Octagone"
            elif corners > 8:
                shape = "Circle"

            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"{shape}: {width_cm}cm x {height_cm}cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# --- Main Execution ---

# Step 1: Calibrate using reference image
ref_image = cv2.imread("reference.png")
marker = find_marker(ref_image)
if marker is None:
    print("Reference object not found!")
    exit()

focal_length = (marker[1][0] * 50) / KNOWN_WIDTH_CM  # Assume distance = 50cm
print(f"Calibrated Focal Length: {focal_length:.2f}")

# Step 2: Real-time video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    marker = find_marker(frame)
    if marker:
        per_width = marker[1][0]
        dist = distance_to_camera(KNOWN_WIDTH_CM, focal_length, per_width)
        pixels_per_cm = per_width / KNOWN_WIDTH_CM

        output = detect_shapes_and_dimensions(frame, pixels_per_cm)
        cv2.putText(output, f"Distance: {round(dist, 2)}cm", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Shape + Measurement", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
