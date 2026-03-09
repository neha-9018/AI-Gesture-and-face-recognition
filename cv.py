import cv2
import face_recognition
import mediapipe as mp

print("Starting AI Gesture Control System...")
print("Press ESC to exit\n")

# ==============================
# Load Authorized Face
# ==============================

authorized_image = face_recognition.load_image_file("authorized.jpg")
authorized_encoding = face_recognition.face_encodings(authorized_image)[0]

# ==============================
# MediaPipe Setup
# ==============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# Start Camera
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

verified = False

# ==============================
# Main Loop
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ==================================
    # FACE VERIFICATION
    # ==================================

    if not verified:

        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encode in encodings:

            match = face_recognition.compare_faces([authorized_encoding], encode)

            if True in match:

                verified = True
                print("Authorized Person Detected")

                cv2.putText(frame,
                            "FACE VERIFIED",
                            (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

            else:

                cv2.putText(frame,
                            "UNAUTHORIZED PERSON",
                            (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            2)

    # ==================================
    # HAND GESTURE CONTROL
    # ==================================

    if verified:

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                finger_tips = [4,8,12,16,20]
                landmarks = hand_landmarks.landmark

                finger_count = 0

                # Thumb
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0]-1].x:
                    finger_count += 1

                # Other fingers
                for tip in finger_tips[1:]:

                    if landmarks[tip].y < landmarks[tip-2].y:
                        finger_count += 1

                # Show finger count
                cv2.putText(frame,
                            f'Fingers: {finger_count}',
                            (20,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,255),
                            2)

                # ==================================
                # ACTIONS
                # ==================================

                if finger_count == 1:
                    action = "Bulb ON"

                elif finger_count == 2:
                    action = "Fan ON"

                elif finger_count == 3:
                    action = "Motor ON"

                elif finger_count == 4:
                    action = "Light OFF"

                elif finger_count == 5:
                    action = "System Stop"

                else:
                    action = "Standby"

                cv2.putText(frame,
                            action,
                            (20,150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                print(action)

    cv2.imshow("AI Control System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# Cleanup
# ==============================

cap.release()
cv2.destroyAllWindows()

print("System Closed")