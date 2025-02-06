import torch
import cv2
import numpy as np
from collections import defaultdict, deque
from lstm_model import LSTMModel
import time  # For timing FPS calculations

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
print(model.names)

# Load trained LSTM model
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load('lstm_model.pth'))
lstm_model.eval()

# Define video paths
video_path = 'traffic.mp4'
cap = cv2.VideoCapture(video_path)

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Input video FPS: {original_fps}")
out = cv2.VideoWriter('output_vehicle_count.avi', cv2.VideoWriter_fourcc(*'XVID'), original_fps, (frame_width, frame_height))

# Counting line position
counting_line_y = 400
vehicles_counted = 0

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Class counts
class_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}

# Object tracking and trajectory storage
tracked_objects = {}  # {object_id: (centroid, already_counted, class)}
trajectories = defaultdict(lambda: deque(maxlen=5))  # {object_id: [centroid_1, ..., centroid_5]}
crossing_confidences = defaultdict(list)  # Store confidence values for each object
next_object_id = 1  # Incremental object ID

# FPS tracking
total_frames_processed = 0
total_processing_time = 0

# Helper functions
def draw_counting_line(frame, y):
    cv2.line(frame, (0, y), (frame.shape[1], y), (255, 255, 0), 2)
    cv2.putText(frame, "Counting Line", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def euclidean_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def normalize_trajectory(trajectory, frame_width, frame_height):
    return [(x / frame_width, y / frame_height) for x, y in trajectory]

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer for FPS calculation
    start_time = time.time()

    # Reduce frame size for efficiency
    frame_resized = cv2.resize(frame, (1600, 1080))
   
    # YOLOv5 detection
    results = model(frame_resized)
    detections = results.pandas().xyxy[0]

    current_frame_objects = {}  # To store objects detected in the current frame

    for _, row in detections.iterrows():
        if row['confidence'] >= CONFIDENCE_THRESHOLD and row['name'] in class_counts.keys():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Match with tracked objects
            matched_id = None
            min_distance = float('inf')

            for obj_id, (prev_centroid, already_counted, vehicle_class) in tracked_objects.items():
                distance = euclidean_distance(centroid, prev_centroid)
                if distance < min_distance and distance < 50:  # Adjust threshold
                    min_distance = distance
                    matched_id = obj_id

            # Assign a new ID if no match is found
            if matched_id is None:
                matched_id = next_object_id
                next_object_id += 1

            # Update current frame objects and trajectory
            current_frame_objects[matched_id] = (centroid, False, row['name'])
            trajectories[matched_id].append(centroid)

            # Extract the confidence score and vehicle type name
            confidence = row['confidence']
            vehicle_name = row['name']
            confidence_text = f"{confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Draw confidence score above the vehicle name
            cv2.putText(frame_resized, confidence_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw vehicle type name below the confidence score
            cv2.putText(frame_resized, vehicle_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Check for line crossing using LSTM or fallback logic
    for obj_id, (centroid, already_counted, vehicle_class) in current_frame_objects.items():
        if obj_id in tracked_objects:
            prev_centroid = tracked_objects[obj_id][0]

            # Use LSTM if sufficient trajectory data is available
            if len(trajectories[obj_id]) == 5:
                normalized_traj = normalize_trajectory(trajectories[obj_id], frame_width, frame_height)
                input_sequence = torch.tensor(normalized_traj, dtype=torch.float32).unsqueeze(0)
                crossing_prediction = lstm_model(input_sequence).item()

                # Add to confidence buffer
                crossing_confidences[obj_id].append(crossing_prediction)
                if len(crossing_confidences[obj_id]) > 5:
                    crossing_confidences[obj_id].pop(0)

                avg_confidence = sum(crossing_confidences[obj_id]) / len(crossing_confidences[obj_id])
                print(f"Object {obj_id} - Prediction: {crossing_prediction:.2f}, Smoothed Confidence: {avg_confidence:.2f}")

                if avg_confidence > 0.5 and not already_counted:
                    if prev_centroid[1] < counting_line_y <= centroid[1]:  # Crossing downward
                        vehicles_counted += 1
                        class_counts[vehicle_class] += 1
                        current_frame_objects[obj_id] = (centroid, True, vehicle_class)
                        print(f"Vehicle ID: {obj_id} Type: {vehicle_class} counted.")

    # Update tracked objects
    tracked_objects = current_frame_objects

    # End timer and calculate FPS for this frame
    elapsed_time = time.time() - start_time
    processing_fps = 1 / elapsed_time if elapsed_time > 0 else 0
    total_processing_time += elapsed_time
    total_frames_processed += 1

    # Draw counting line and display totals
    draw_counting_line(frame_resized, counting_line_y)
    cv2.putText(frame_resized, f"Total Vehicles: {vehicles_counted}", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 255, 0), 2)

    # Display class-wise counts
    y_offset = 100
    for vehicle_class, count in class_counts.items():
        cv2.putText(frame_resized, f"{vehicle_class.capitalize()}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 2)
        y_offset += 30

    # Write to output
    out.write(frame_resized)
    cv2.imshow('Vehicle Counting', frame_resized)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average FPS
average_fps = total_frames_processed / total_processing_time
print(f"Average Processing FPS: {average_fps:.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()