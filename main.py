# Import libraries we need
import cv2  # For camera and image processing
import mediapipe as mp  # For detecting hands
import pyautogui  # For controlling the mouse cursor
import numpy as np  # For math calculations
import time  # For timing delays

# Set up hand detection from MediaPipe
hand_detector = mp.solutions.hands.Hands(
    static_image_mode=False,  # Process video stream (not single images)
    max_num_hands=1,  # Only detect one hand at a time
    model_complexity=0,  # Use faster but less accurate model
    min_detection_confidence=0.7,  # How confident we need to be to detect a hand
    min_tracking_confidence=0.6  # How confident we need to be to track the hand
)
drawing_utils = mp.solutions.drawing_utils  # For drawing hand landmarks on screen

# Get the size of your computer screen
screen_width, screen_height = pyautogui.size()

# Set up the camera
cap = cv2.VideoCapture(0)  # Use the first camera (usually built-in webcam)
cap.set(3, 640)  # Set camera width to 640 pixels
cap.set(4, 480)  # Set camera height to 480 pixels

# Variables to make mouse movement smooth (not jumpy)
prev_x, prev_y = 0, 0  # Remember where the mouse was last time
smooth_factor = 0.2  # How much smoothing to apply (lower = smoother)

# Settings for click detection
click_threshold = 25  # How close fingers need to be to register a click
click_delay = 0.5  # Wait this long between clicks (in seconds)
last_left_click = 0  # When we last did a left click
last_double_click = 0  # When we last did a double click
double_click_interval = 0.4  # How fast you need to click twice for double-click
last_right_click = 0  # When we last did a right click

# Keep track of whether we're currently clicking (prevents multiple clicks)
left_clicking = False
right_clicking = False


def get_position(landmark, frame_w, frame_h):
    """
    Convert hand landmark position to pixel coordinates on the camera frame
    landmark: A single point from the hand detection
    frame_w, frame_h: Width and height of the camera frame
    Returns: x and y coordinates in pixels
    """
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


try:
    # Main loop - runs continuously until you press ESC
    while True:
        # Get a frame from the camera
        success, frame = cap.read()
        if not success:
            break  # Stop if camera fails

        # Flip the image horizontally (like a mirror)
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Convert image colors for hand detection (MediaPipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Look for hands in the frame
        result = hand_detector.process(rgb_frame)

        action = "Idle"  # What action we're performing (shown on screen)

        # If we found a hand
        if result.multi_hand_landmarks:
            # Get the first hand found
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw the hand skeleton on the video
            drawing_utils.draw_landmarks(frame, hand_landmarks)

            # Get all the landmark points
            lm = hand_landmarks.landmark

            # Get the positions of important fingertips
            # Each number corresponds to a specific point on the hand
            index_x, index_y = get_position(lm[8], frame_width, frame_height)  # Index finger tip
            thumb_x, thumb_y = get_position(lm[4], frame_width, frame_height)  # Thumb tip
            middle_x, middle_y = get_position(lm[12], frame_width, frame_height)  # Middle finger tip

            # Draw colored circles on the fingertips so you can see them
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), -1)  # Yellow circle on index
            cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), -1)  # Purple circle on thumb
            cv2.circle(frame, (middle_x, middle_y), 8, (0, 0, 255), -1)  # Red circle on middle

            # Convert index finger position to screen coordinates
            # Map from camera frame (0 to 1) to full screen size
            target_x = np.interp(lm[8].x, [0, 1], [0, screen_width])
            target_y = np.interp(lm[8].y, [0, 1], [0, screen_height])

            # Make mouse movement smooth (not jerky)
            # Mix old position with new position
            curr_x = prev_x * (1 - smooth_factor) + target_x * smooth_factor
            curr_y = prev_y * (1 - smooth_factor) + target_y * smooth_factor

            # Actually move the mouse cursor
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            prev_x, prev_y = curr_x, curr_y  # Remember this position for next time

            # Calculate distances between fingertips
            index_thumb_dist = np.hypot(index_x - thumb_x, index_y - thumb_y)  # Distance between index and thumb
            middle_thumb_dist = np.hypot(middle_x - thumb_x, middle_y - thumb_y)  # Distance between middle and thumb
            current_time = time.time()  # Get current time

            # LEFT CLICK and DOUBLE CLICK detection
            # When index finger and thumb are close together
            if index_thumb_dist < click_threshold:
                # Only click if we're not already clicking and enough time has passed
                if not left_clicking and (current_time - last_left_click) > click_delay:
                    # Check if this might be a double click
                    # (if we clicked recently, this could be the second click)
                    if (current_time - last_double_click) < double_click_interval:
                        pyautogui.doubleClick()  # Perform double click
                        action = "Double Click"
                        last_double_click = 0  # Reset the double click timer
                    else:
                        # Regular single click
                        pyautogui.click()
                        action = "Left Click"
                        last_left_click = current_time
                        last_double_click = current_time  # Start double click timer
                    left_clicking = True  # Mark that we're clicking
            else:
                left_clicking = False  # Fingers are apart, not clicking anymore

            # RIGHT CLICK detection
            # When middle finger and thumb are close together
            if middle_thumb_dist < click_threshold:
                # Only click if we're not already clicking and enough time has passed
                if not right_clicking and (current_time - last_right_click) > click_delay:
                    pyautogui.click(button='right')  # Perform right click
                    action = "Right Click"
                    right_clicking = True  # Mark that we're right clicking
                    last_right_click = current_time
            else:
                right_clicking = False  # Fingers are apart, not right clicking anymore

            # If no clicking happened, we're just moving the mouse
            if action == "Idle":
                action = "Move"

        # Show what action we're doing on the video
        cv2.putText(frame, f'Action: {action}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video window
        cv2.imshow("Virtual Mouse", frame)

        # Check if ESC key is pressed to quit
        if cv2.waitKey(1) == 27:  # 27 is the ESC key code
            break

except KeyboardInterrupt:
    # Handle when user presses Ctrl+C
    print("Interrupted by user")
except Exception as e:
    # Handle any other errors
    print(f"Error occurred: {e}")
finally:
    # Clean up when program ends
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows
