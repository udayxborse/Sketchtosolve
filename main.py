import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Streamlit UI setup
st.set_page_config(layout="wide")
st.image('MathGestures.png')

# Chat history section
st.title("AI Math Problem Solver")

# AI configuration
genai.configure(api_key="AIzaSyC6_7-hxVqljm-FD8-J6DCPRZsrQO7SzJM")
model = genai.GenerativeModel('gemini-1.5-flash')

# Feature 1: Hand Gesture Recognition setup
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run Hand Gesture Mode', value=True)
    FRAME_WINDOW = st.empty()
    # Area for displaying answers
    answer_area = st.empty()  # To show AI responses

with col2:
    st.title("Upload and Solve")
    st.subheader("Upload an image with a math problem")
    uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# AI response storage
output_texts = []  # Store answers from both features

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

prev_pos = None
canvas = None

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up to clear canvas
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Gesture to solve math problem
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text

# Handle uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Solve Math Problem from Image"):
        image = Image.open(uploaded_image)
        response = model.generate_content(["Solve this math problem", image])
        output_texts.append("AI: " + response.text)  # Store the AI response
        answer_area.markdown("\n\n".join(output_texts))  # Display AI response in the answer area

# Input area for chatting with AI
user_input = st.text_input("Type your message here:", "")
send_button = st.button("Send")

# Handle user input to interact with AI
if send_button and user_input:
    response = model.generate_content([user_input])
    output_texts.append("You: " + user_input)  # Add user input to chat history
    output_texts.append("AI: " + response.text)  # Add AI response to chat history
    answer_area.markdown("\n\n".join(output_texts))  # Display updated chat history in answer area

# Main loop for Feature 1: Hand Gesture Recognition
while run:
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = np.zeros_like(img)
        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            output_text = sendToAI(model, canvas, fingers)
            if output_text:
                output_texts.append("AI: " + output_text)  # Store the AI response
                answer_area.markdown("\n\n".join(output_texts))  # Display updated chat history in the answer area

        # Overlay canvas and display
        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")
    else:
        st.write("Failed to capture image")

# Release the webcam when done
cap.release()
