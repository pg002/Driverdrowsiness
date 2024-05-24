import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os
import time
import pygame.mixer as mixer

# Initialize the mixer and load the alert sound
mixer.init()
mixer.music.load("music.wav")

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def eye_aspect_ratio(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR) to detect drowsiness."""
    # Calculate the Euclidean distances between the sets of vertical eye landmarks
    vertical_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Calculate the Euclidean distance between the set of horizontal eye landmarks
    horizontal_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    # Calculate the Eye Aspect Ratio (EAR)
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """Calculate the Mouth Aspect Ratio (MAR) to detect yawning."""
    # Calculate the Euclidean distances between the sets of vertical mouth landmarks
    vertical_dist1 = euclidean_distance(mouth_landmarks[9], mouth_landmarks[10])
    vertical_dist2 = euclidean_distance(mouth_landmarks[11], mouth_landmarks[12])
    vertical_dist3 = euclidean_distance(mouth_landmarks[13], mouth_landmarks[14])
    
    # Calculate the Euclidean distance between the set of horizontal mouth landmarks
    horizontal_dist = euclidean_distance(mouth_landmarks[0], mouth_landmarks[1])

    # Calculate the Mouth Aspect Ratio (MAR)
    mar = (vertical_dist1 + vertical_dist2 + vertical_dist3) / (3.0 * horizontal_dist)
    return mar

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Drawing specifications for eye landmarks
draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
frame_count = 0
flag = 0
alert_flag = 0

# Initialize variables for FPS calculation
start_time = time.time()
frame_counter = 0
fps=0

# Ensure the directory for saved frames exists
os.makedirs('saved_frames_mediapipe', exist_ok=True)

# Latency calculation variables
latency_values = []

# Start capturing video
with mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_frame_time = time.time()

        # Convert the image to RGB and process it with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Extract face landmarks if available
        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks for the left eye 
                left_eye_indices = [362, 386, 387, 263, 373,374]
                left_eye_landmarks = []
                for i in left_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    left_eye_landmarks.append((x, y))
                
                # Extract landmarks for the right eye
                right_eye_indices = [33, 159, 158, 133, 153,145]
                right_eye_landmarks = []
                for i in right_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    right_eye_landmarks.append((x, y))

                left_ear = eye_aspect_ratio(left_eye_landmarks)
                right_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_ear + right_ear) / 2.0

                # Extract landmarks for the mouth
                mouth_indices = [78,308,191,95,80,88,81,178,82,87,13,14,312,317,311,402,310,318,415,324]
                mouth_landmarks = []
                for i in mouth_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    mouth_landmarks.append((x, y))

                mar = mouth_aspect_ratio(mouth_landmarks)

                # for (x, y) in left_eye_landmarks + right_eye_landmarks + mouth_landmarks:
                #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Draw landmarks on the face
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                # Check for drowsiness based on EAR
                if ear <= 0.25:
                    alert_flag +=1
                    if alert_flag >= 10:
                        end_frame_time = time.time()
                        frame_latency = end_frame_time - start_frame_time
                        latency_values.append(frame_latency)
                        cv2.putText(frame, "Alert", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                
                elif ear <= 0.29 or mar >= 0.60:
                    flag += 1
                    if flag >= 10:
                        end_frame_time = time.time()
                        frame_latency = end_frame_time - start_frame_time
                        latency_values.append(frame_latency)
                        cv2.putText(frame, "Warning", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        # mixer.music.play()
                else:
                    flag = 0
                    alert_flag=0
                    mixer.music.stop()

                # Display the EAR and MAR values
            
                # cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the frame
        cv2.imwrite(f"saved_frames_mediapipe/frame_{frame_count}.png", frame)
        frame_count += 1

        # Display FPS
        frame_counter += 1
        if time.time() - start_time >= 1:
            fps = frame_counter / (time.time() - start_time)
            frame_counter = 0
            start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Eye and Mouth Aspect Ratio', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
# Plot the latency graph
plt.figure()
plt.plot(latency_values)
plt.xlabel("Frame")
plt.ylabel("Latency (s)")
plt.title("Software Latency")
plt.show()
