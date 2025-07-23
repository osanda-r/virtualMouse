import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe
hand_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
drawing_utils = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Start camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width (optional: reduce for better FPS)
cap.set(4, 480)  # Height

# Initial values
prev_x, prev_y = 0, 0
smooth_factor = 0.2  # Between 0 (instant) and 1 (very slow)

click_threshold = 25
click_delay = 0.5  # seconds
last_click_time = 0
clicking = False

def get_position(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)

    action = "Idle"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        drawing_utils.draw_landmarks(frame, hand_landmarks)

        lm = hand_landmarks.landmark
        index_x, index_y = get_position(lm[8], frame_width, frame_height)
        thumb_x, thumb_y = get_position(lm[4], frame_width, frame_height)

        # Draw fingertips
        cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), -1)

        # Map to screen coords
        target_x = np.interp(lm[8].x, [0, 1], [0, screen_width])
        target_y = np.interp(lm[8].y, [0, 1], [0, screen_height])

        # Apply smoothing (exponential moving average)
        curr_x = prev_x * (1 - smooth_factor) + target_x * smooth_factor
        curr_y = prev_y * (1 - smooth_factor) + target_y * smooth_factor
        pyautogui.moveTo(curr_x, curr_y, duration=0)

        prev_x, prev_y = curr_x, curr_y

        # Click logic
        distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
        if distance < click_threshold:
            if not clicking and (time.time() - last_click_time) > click_delay:
                pyautogui.click()
                clicking = True
                last_click_time = time.time()
                action = "Click"
        else:
            clicking = False
            action = "Move"

    # Show status text
    cv2.putText(frame, f'Action: {action}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Virtual Mouse", frame)

    # Exit key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
