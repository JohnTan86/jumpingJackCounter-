import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
jump_count = 0
last_jump = False

# Define circle properties for the reset button
circle_radius = 30
circle_center = (640 - circle_radius - 10, circle_radius + 10)  # Assuming 640 is the width of the frame

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    line_y = h // 3  # Horizontal line halfway through the image
    cv2.line(image, (0, line_y), (w, line_y), (255, 0, 0), 2)  # Draw the line

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmarks for shoulders and hands
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

        # Check if shoulders pass the line
        shoulders_above_line = left_shoulder.y * h < line_y and right_shoulder.y * h < line_y

        # Check if hands meet above the head
        hands_meet_above = (left_hand.y * h < left_shoulder.y * h and right_hand.y * h < right_shoulder.y * h and
                            abs(left_hand.x - right_hand.x) * w < 0.1 * w)
        # Check if either hand is in the reset circle
        left_hand_pos = (int(left_hand.x * w), int(left_hand.y * h))
        right_hand_pos = (int(right_hand.x * w), int(right_hand.y * h))
        if (np.linalg.norm(np.array(left_hand_pos) - np.array(circle_center)) < circle_radius or
            np.linalg.norm(np.array(right_hand_pos) - np.array(circle_center)) < circle_radius):
            jump_count = 0  # Reset the count
            
        # Counting logic
        if shoulders_above_line and hands_meet_above:
            if not last_jump:
                jump_count += 1
                last_jump = True
        else:
            last_jump = False

        cv2.putText(image, f'Jump Count: {jump_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Jumping Jack Counter', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
