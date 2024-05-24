import os
import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure the directory for saving faces exists
if not os.path.exists("saved_faces_dlib"):
    os.makedirs("saved_faces_dlib")

# Ensure the directory for saving frames exists
if not os.path.exists("saved_frames_dlib"):
    os.makedirs("saved_frames_dlib")

# Initialize the mixer and load the alert sound
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (2.0 * D)

# Thresholds
ear_warning_thresh = 0.28
ear_alert_thresh = 0.25
mouth_thresh = 0.50
blink_thresh = 0.25
frame_check = 10
blink_check_frames = 4
blink_check_time = 0.2  # in seconds

fps = 0
# Initialize dlib's face detector 
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]

ear_dict = {"time": [], "ear": []}
mar_dict = {"time": [], "mar": []}

# Start the video stream
cap = cv2.VideoCapture(0)
flag = 0
alert_flag = 0
image_count = 0 
start = time.time()

# FPS calculation variables
fps_start_time = time.time()
frame_count = 0

# Blinking variables
blink_start_time = None
blink_frame_counter = 0

# Latency calculation variables
latency_values = []


while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Measure the start time of frame processing
    frame = imutils.resize(frame, width=600)
    start_frame_time = time.time()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    gray = cv2.equalizeHist(gray)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(mouth)

        # Average the EAR
        ear = (leftEAR + rightEAR) / 2.0

        # ear_values.append(ear)
        # mar_values.append(mouthEAR)
        ear_dict["ear"].append(ear)
        ear_dict["time"].append(time.time() - start)

        mar_dict["mar"].append(mouthEAR)
        mar_dict["time"].append(time.time() - start)

        # Visualize the eye landmarks
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check for continuous blinking
        if ear < blink_thresh:
            if blink_start_time is None:
                blink_start_time = time.time()
            blink_frame_counter += 1
        else:
            if blink_start_time is not None:
                blink_duration = time.time() - blink_start_time
                if blink_duration <= blink_check_time and blink_frame_counter >= blink_check_frames:
                    cv2.putText(frame, "Blinking", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
                    # mixer.music.play()
                blink_start_time = None
                blink_frame_counter = 0

        if ear <= ear_alert_thresh:
            alert_flag += 1
            if alert_flag >= frame_check:
                mixer.music.play()
                end_frame_time = time.time()
                frame_latency = end_frame_time - start_frame_time
                latency_values.append(frame_latency)
                cv2.putText(
                    frame,
                    "Alert",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        elif (ear < ear_warning_thresh or mouthEAR >= mouth_thresh):
            flag += 1
            if flag >= frame_check:
                end_frame_time = time.time()
                frame_latency = end_frame_time - start_frame_time
                latency_values.append(frame_latency/1000)
                cv2.putText(
                    frame,
                    "Warning",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
        else:
            mixer.music.stop()
            flag = 0
            alert_flag = 0
        # Display EAR and MAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mouthEAR:.2f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # FPS calculation
    frame_count += 1
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time

    if time_diff >= 1:
        fps = frame_count / time_diff
        fps_start_time = time.time()
        frame_count = 0

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with the eye aspect ratio
    cv2.imshow("Frame", frame)

    # Saving Frame in Local
    cv2.imwrite(f"saved_frames_dlib/frame_{image_count}.png", frame)
    image_count += 1

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Plot the EAR over time
plt.figure()
# plt.plot(ear_values)
plt.plot(ear_dict["time"], ear_dict["ear"], label="EAR")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.title("Eye Aspect Ratio over Time")
plt.legend()
plt.show()

# # Plot the MAR over time
plt.figure()
plt.plot(mar_dict["time"], mar_dict["mar"], label="MAR")
# plt.plot(mar_values)
plt.xlabel("Time (s)")
plt.ylabel("MAR")
plt.title("Mouth Aspect Ratio over Time")
plt.legend()
plt.show()

# Plot the latency graph
plt.figure()
plt.plot(latency_values)
plt.xlabel("Frame")
plt.ylabel("Latency (s)")
plt.title("Software Latency")
plt.show()
