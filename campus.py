import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech
engine = pyttsx3.init()

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected. Exiting...")
    exit()

# Load face and QR detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
qr_detector = cv2.QRCodeDetector()

# ---- Broad HSV ranges for robust detection ----
green_lower = np.array([35, 50, 50])
green_upper = np.array([85, 255, 255])

blue_lower = np.array([90, 50, 50])
blue_upper = np.array([140, 255, 255])

print("✅ System Ready: Show your face and ID card near chest (Green = Blind, Blue = Deaf, or QR)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected or frame not received. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Define region below face (ID card area)
        id_region_y1 = y + int(1.2 * h)
        id_region_y2 = y + int(2.5 * h)
        id_region_x1 = x - int(0.3 * w)
        id_region_x2 = x + int(1.3 * w)

        # Clamp within frame size
        id_region_y1 = max(0, id_region_y1)
        id_region_y2 = min(frame.shape[0], id_region_y2)
        id_region_x1 = max(0, id_region_x1)
        id_region_x2 = min(frame.shape[1], id_region_x2)

        if id_region_x2 <= id_region_x1 or id_region_y2 <= id_region_y1:
            continue

        id_region = frame[id_region_y1:id_region_y2, id_region_x1:id_region_x2]
        if id_region is None or id_region.size == 0:
            continue

        # Convert to HSV
        hsv_region = cv2.cvtColor(id_region, cv2.COLOR_BGR2HSV)
        hsv_region = cv2.GaussianBlur(hsv_region, (5, 5), 0)

        # Draw ID region box
        cv2.rectangle(frame, (id_region_x1, id_region_y1),
                      (id_region_x2, id_region_y2), (255, 255, 0), 2)
        cv2.putText(frame, "ID region", (id_region_x1, id_region_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Create color masks
        green_mask = cv2.inRange(hsv_region, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv_region, blue_lower, blue_upper)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # Calculate detected area size
        green_area = cv2.countNonZero(green_mask)
        blue_area = cv2.countNonZero(blue_mask)

        # ---- QR Detection ----
        qr_data, points, _ = qr_detector.detectAndDecode(id_region)
        if qr_data:
            qr_data = qr_data.lower()
            cv2.putText(frame, f"QR Detected: {qr_data}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if "blind" in qr_data:
                cv2.putText(frame, "QR: Blind User - Voice Mode", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                engine.say("QR recognized: User is blind. Voice guidance activated.")
                engine.runAndWait()
            elif "deaf" in qr_data:
                cv2.putText(frame, "QR: Deaf User - Text Mode", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # No voice output for deaf users
            else:
                cv2.putText(frame, f"QR Info: {qr_data}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                engine.say(f"QR recognized: {qr_data}. Welcome to campus.")
                engine.runAndWait()
            break

        # ---- Compare color areas and detect based on larger area ----
        if green_area > 1500 or blue_area > 1500:
            if green_area > blue_area:
                cv2.putText(frame, "Blind ID Detected (Green)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                engine.say("Welcome! Voice guidance activated.")
                engine.runAndWait()
            else:
                cv2.putText(frame, "Deaf ID Detected (Blue)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, "Welcome! Text assistance activated.",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # No voice output for deaf users
            break

    # Display main camera feed only
    cv2.imshow("AI CampusSense - Smart Detection", frame)

    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("Exiting system...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
