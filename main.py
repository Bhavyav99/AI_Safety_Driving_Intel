import cv2
import streamlit as st
from gaze_tracking import GazeTracking
import time
import numpy as np
from PIL import Image
from playsound import playsound

# from pyVHR.analysis.pipeline import Pipeline
# import matplotlib.pyplot as plt


# pipe = Pipeline()
# time, BPM, uncertainty = pipe.run_on_video('movie.mov', roi_approach="patches", roi_method="faceparsing", cuda=False) # type: ignore
# # st.write(time, BPM, uncertainty)
# plt.figure()
# plt.plot(time, BPM)
# plt.fill_between(time, BPM-uncertainty, BPM+uncertainty, alpha=0.2)
# print(time, BPM, uncertainty)
# st.pyplot(plt.gcf())  # Display the plot in Streamlit

from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors

def run_pyVHR_pipeline():
    # Set up loading screen
    with st.spinner('Running pyVHR pipeline...'):
        # params
        roi_approach = 'patches'   # 'holistic' or 'patches'
        bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
        method = 'cpu_CHROM'       # one of the methods implemented in pyVHR
        
        # run
        pipe = Pipeline()          # object to execute the pipeline
        time_t, BPM, uncertainty = pipe.run_on_video('/Users/knowhrishi/Documents/video.mov',
                                            roi_method='convexhull',
                                            roi_approach=roi_approach,
                                            method=method,
                                            pre_filt=True,
                                            post_filt=True,
                                            cuda=False, 
                                            verb=True)
    
    # Display results
    st.success('pyVHR pipeline finished!')
    # Use the visualize module of pyVHR to plot the results
    
    return(time, BPM, uncertainty)

# plt.figure()
# plt.plot(time, BPM)
# plt.fill_between(time, BPM-uncertainty, BPM+uncertainty, alpha=0.2)
# print(time, BPM, uncertainty)
# st.pyplot(plt.gcf())  # Display the plot in Streamlit

st.set_page_config(page_title="AI Guardian Angel")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpeg') 
# Load the pre-trained cascade classifier for hand gestures
hand_cascade = cv2.CascadeClassifier('hand_model.xml')
# Initialize hand count to 0
hand_count = 0

title_color = "#FFFF"  # Replace with your desired color code
font_sizet = "30px"
font_familyt = "DS-Digital, sans-serif"
font_size = "50px"
font_family = "Trebuchet MS, sans-serif"

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
warning_logo = Image.open("warning.png")
warning_logo = warning_logo.resize((100, 100))

title = st.empty()
title.markdown(f"<p style='font-size:{font_size}; font-family:{font_family}; color:{title_color}'>AI Guardian Angel</p>", unsafe_allow_html=True)

with st.sidebar:
    webcam = st.checkbox("Start AI Guardian")
    volume_enabled = st.checkbox("Enable volume control")
    pyVHR_check = st.checkbox("Enable pyVHR")

# st.title("AI Guardian Angel")

blink_count = 0
blink_text = ""
start_time = time.time()
warning_time = 1  # Set the warning time to 1 second

right_time = 0  # Set the time the user has been looking right to 0
warning_message = ""  # Initialize the warning message to an empty string
# warning_placeholder = st.empty()  # Create a placeholder for the warning message
container = st.container()  # Create a container to align elements to the left
with container:
    col1, col2 = st.columns(2)  # Create two columns aligned to the left
    with col1:
        stframe = st.empty()  # Put the video feed in the left column
    with col2:
        warning_placeholder = st.empty()  # Create a placeholder for the warning message in the right column


logtxt = 75

t = st.empty()
counter = 1

if webcam:
    webcam = cv2.VideoCapture(0)
    # playsound('startsound.mp3')
    col1, col2, col3, col4= st.columns(4)  # Create two columns
    with col1:
        stframe = st.empty()  # Put the video feed in the left column
    with col2:
        warning_placeholder = st.empty()  # Create a placeholder for the warning message in the right column
    with col3:
        text1 = st.empty()  # Create the first additional text box in the third column
    with col4:
        text2 = st.empty()  # Create the second additional text box in the fourth column
        text3 = st.empty()  # Create the third additional text box in the fourth column


    while True:
        ret, frame = webcam.read()
        if not ret:
            st.error("Failed to read frame from webcam")
            break
        # Add button to run pyVHR pipeline
        

        if frame is not None:
            gaze.refresh(frame)
            frame = gaze.annotated_frame()
            text = ""
            # Convert the frame to grayscale and detect hand gestures
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

            # If a hand gesture is detected, print a message to the console
            if volume_enabled:
                if len(hands) >= 0:
                    logtxt -= 1
                    t.write('VOLUME: ' + str(logtxt))
                    t.markdown(f"<p style='font-size:{font_sizet}; font-family:{font_familyt}; color:{title_color}'>VOLUME: {logtxt}</p>", unsafe_allow_html=True)
            if pyVHR_check:
                time_t, BPM_t, uncertainty = run_pyVHR_pipeline()
                # text1.text("Time: **"+ time)
                # text2.text("**BPM: **" + round(BPM))
                # text3.text("**uncertainty: **"+ uncertainty)
                plt.figure()
                plt.plot(time_t, BPM_t)
                plt.fill_between(time_t, BPM_t-uncertainty, BPM_t+uncertainty, alpha=0.2)
                # print(time, BPM, uncertainty)
                # st.pyplot(plt.gcf())  
                plt.savefig('pyVHR.png')

            if gaze.is_blinking():
                blink_count += 1
                if blink_count == 1:
                    start_time = time.time()  # Start the timer when the first blink is detected
                elif time.time() - start_time >= warning_time:
                    blink_text = "Eyes closed"
                    # Add the warning logo to the frame
                    warning_image = cv2.cvtColor(np.array(warning_logo), cv2.COLOR_RGB2BGR)
                    x_offset = 10
                    y_offset = 10
                    frame[y_offset:y_offset+warning_image.shape[0], x_offset:x_offset+warning_image.shape[1]] = warning_image
                    warning_placeholder.warning(blink_text)  # Show the warning message once the timer reaches 3 seconds

            else:
                blink_count = 0
                blink_text = ""
                start_time = time.time()  # Reset the timer when no blink is detected
                warning_placeholder.empty()  # Hide the warning message

            if time.time() - start_time >= warning_time and blink_count >= warning_time:
                blink_text = "Eyes closed"
                # Add the warning logo to the frame
                warning_image = cv2.cvtColor(np.array(warning_logo), cv2.COLOR_RGB2BGR)
                x_offset = 10
                y_offset = 10
                frame[y_offset:y_offset+warning_image.shape[0], x_offset:x_offset+warning_image.shape[1]]= warning_image
                warning_placeholder.warning(blink_text)  # Show the warning message once the timer reaches 3 seconds

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) == 27:
            break

# webcam.release()
cv2.destroyAllWindows()