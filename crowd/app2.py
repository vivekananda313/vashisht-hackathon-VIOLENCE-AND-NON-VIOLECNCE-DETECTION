import cv2
import numpy as np
from datetime import datetime
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

# Load pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained Keras model for violence detection
model = load_model(r'modelnew.h5')

# Define the sequence length
SEQUENCE_LENGTH = 10

# Function to preprocess a single frame
def preprocess_frame(frame):
    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize the frame to match the model's input shape
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    # Normalize pixel values to the range [0, 1]
    frame /= 255.0
    return frame

# Function to preprocess a sequence of frames
def preprocess_sequence(frames):
    # Concatenate and preprocess the frames
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    return np.array(preprocessed_frames)

# Function to predict violence given a sequence of frames
def predict_violence(sequence):
    # Make prediction
    prediction = model.predict(sequence)
    confidence = np.max(prediction)
    label = "Violence" if prediction[0][0] > 0.8 else "Non-violence"
    return label, confidence

# Function to send email
def send_email(subject, message, image, time):
    sender_email = "yourmail"
    password = "password"
    receiver_email = "receiver_email"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    text = f"Violence is happening! Please take immediate action. Detected at {time}."
    msg.attach(MIMEText(text, 'plain'))

    with open(image, 'rb') as fp:
        img = MIMEImage(fp.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image))
        msg.attach(img)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Open webcam for capturing video
cap = cv2.VideoCapture(0)

# Variables to store frames for the sequence
frame_sequence = []

# Counter to keep track of the total number of violent frames
total_violent_frames = 0

# Create directory to save images if not exists
save_dir = 'images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Append frame to the sequence
    frame_sequence.append(frame)

    # Keep the sequence length constant
    if len(frame_sequence) > SEQUENCE_LENGTH:
        frame_sequence.pop(0)

    # If the sequence has reached the desired length, preprocess and predict
    if len(frame_sequence) == SEQUENCE_LENGTH:
        # Preprocess the sequence
        preprocessed_sequence = preprocess_sequence(frame_sequence)

        # Predict violence for the entire sequence
        frame_prediction, confidence = predict_violence(preprocessed_sequence)

        # Display frame-level violence prediction and confidence
        if confidence>0.8:
            box_color = (0, 0, 255)
            cv2.putText(frame, f"Violence - Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
            # If violence is detected and less than 5 violent frames have been sent, save the frame as an image and send it via email
            if confidence>0.8:
                total_violent_frames += 1
                if total_violent_frames == 50:  # Adjust the number of frames to your requirement
                    image_path = os.path.join(save_dir, f"violence_{current_time}.jpg")
                    cv2.imwrite(image_path, frame)
                    send_email("Violence Detected", "Please find the attached image.", image_path, current_time)
        else:
            # Display Non-violence text
            cv2.putText(frame, "Non-violence", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display face detection and crowd monitoring information at the top-middle
    cv2.putText(frame, "Face Detection and Crowd Monitoring System", (int(frame.shape[1] / 2) - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

    # Display the current time
    cv2.putText(frame, current_time, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture object and close all windows
cap.release()
cv2.destroyAllWindows()
