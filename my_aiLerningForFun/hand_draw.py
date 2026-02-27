import cv2
import mediapipe as mp
import numpy as np

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------- Canvas ----------
canvas = None
prev_x, prev_y = 0, 0
brush_color = (0, 0, 255)   # Red
brush_size = 5

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip & pip
            x = int(hand.landmark[8].x * frame.shape[1])
            y = int(hand.landmark[8].y * frame.shape[0])
            y_pip = int(hand.landmark[6].y * frame.shape[0])

            # Middle finger (for clear screen)
            y_middle_tip = int(hand.landmark[12].y * frame.shape[0])
            y_middle_pip = int(hand.landmark[10].y * frame.shape[0])

            # 👉 Draw when index finger is up
            if y < y_pip and y_middle_tip > y_middle_pip:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    brush_color,
                    brush_size
                )
                prev_x, prev_y = x, y

            # ✌️ Clear screen (index + middle up)
            elif y < y_pip and y_middle_tip < y_middle_pip:
                canvas[:] = 0
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    # Combine camera + canvas
    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "Index: Draw | Index+Middle: Clear | Q: Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Hand Drawing AI", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
