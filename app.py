import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

class PushupProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for natural screen view
         
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
         
        try:
            landmarks = results.pose_landmarks.landmark
             
            # Using standard left-side body tracking points
            s, e, w, h, a = 11, 13, 15, 23, 27
             
            shoulder = [landmarks[s].x, landmarks[s].y]
            elbow = [landmarks[e].x, landmarks[e].y]
            wrist = [landmarks[w].x, landmarks[w].y]
            hip = [landmarks[h].x, landmarks[h].y]
            ankle = [landmarks[a].x, landmarks[a].y]
             
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            body_angle = calculate_angle(shoulder, hip, ankle)
             
            x_dist = abs(shoulder[0] - hip[0])
            y_dist = abs(shoulder[1] - hip[1])
             
            if y_dist > x_dist:
                feedback = "Get in Position"
                box_color = (100, 100, 100) # Gray
            else:
                if body_angle > 165:
                    feedback = "Good Form"
                    box_color = (0, 255, 0) # Green
                    if elbow_angle > 160: self.stage = "up"
                    if elbow_angle < 90 and self.stage == "up":
                        self.stage = "down"
                        self.counter += 1
                else:
                    feedback = "Straighten Back!"
                    box_color = (0, 0, 255) # Red
             
            # Draw the clean web UI top bar
            cv2.rectangle(img, (0, 0), (640, 50), box_color, -1)
            status_text = f"REPS: {self.counter} | {feedback}"
            cv2.putText(img, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
             
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        except:
            pass
             
        return frame.from_ndarray(img, format="bgr24")

# Streamlit Interface Configuration
st.set_page_config(page_title="Realtime AI Spotter", layout="centered")
st.title("Realtime AI Spotter 👾")
st.markdown("Set your camera up sideways to analyze your push-up geometry.")

# Launch browser webstream with STUN configuration to bypass network firewalls
webrtc_streamer(
    key="ai-spotter", 
    video_processor_factory=PushupProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
