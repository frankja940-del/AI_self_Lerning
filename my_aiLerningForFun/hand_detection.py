import cv2
import mediapipe as mp

# init mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# จุด (landmark)
dot_style = mp_draw.DrawingSpec(
    color=(255, 224, 159),   # (BGR)
    thickness=6,
    circle_radius=4     # ขนาดจุด
)

# เส้น (connections)
line_style = mp_draw.DrawingSpec(
    color=(255, 131, 248),   # (BGR)
    thickness=3
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("🖐 Hand detection started")
print("Press 'q' or 'ESC' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
           mp_draw.draw_landmarks(
               frame,
               hand_landmarks,
               mp_hands.HAND_CONNECTIONS,
               dot_style,
               line_style
           )

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or ESC
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed")
