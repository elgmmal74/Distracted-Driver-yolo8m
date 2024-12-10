import os
from ultralytics import YOLO
import cv2
import pygame
import torch
import time

# Initialize Pygame for sound
pygame.init()
pygame.mixer.music.load("alarm.wav")

# Load the YOLO model
model = YOLO("bestDriver.pt")

# Ensure GPU is used if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open video file
cap = cv2.VideoCapture("video1.mp4")

Classes = model.names


def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color,
                              thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  background_color,
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  border_color,
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


# Frame skipping for faster processing
frame_skip = 2  # Process every 2nd frame
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video feed is not available or finished.")
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    # Resize frame to reduce processing time
    input_frame = cv2.resize(frame, (640, 360))

    # Perform YOLO prediction
    start_time = time.time()
    results = model.predict(input_frame, conf=0.4, device=device, imgsz=(640, 360))

    # Process results
    for result in results:
        Boxes = result.boxes.xyxy.cpu().numpy()
        Labels = result.boxes.cls.cpu().numpy()
        Confs = result.boxes.conf.cpu().numpy()
        for box, label, conf in zip(Boxes, Labels, Confs):
            x, y, w, h = map(int, box)
            label = int(label)
            conf = float(conf)
            color = (0, 255, 0) if label == 0 else (0, 0, 255)

            # Draw detection boxes and labels
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            draw_text_with_background(frame,
                                      f"{Classes[label].capitalize()} {conf * 100:.2f}%",
                                      (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      color)
            # Play alarm for non-class 0 detections
            if label != 0 and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    draw_text_with_background(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0), (0, 255, 0))

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
