import cv2
import streamlit as st
from gaze_tracking import GazeTracking
import time
from openvino.inference_engine import IECore

# Load the pre-trained driver action recognition model
ie = IECore()
model_path = "/path/to/pretrained/model"
encoder_path = model_path + ".encoder"
decoder_path = model_path + ".decoder"
encoder_net = ie.read_network(model=encoder_path, weights=encoder_path.replace('.xml', '.bin'))
decoder_net = ie.read_network(model=decoder_path, weights=decoder_path.replace('.xml', '.bin'))
encoder_exec = ie.load_network(network=encoder_net, device_name="CPU")
decoder_exec = ie.load_network(network=decoder_net, device_name="CPU")

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

st.title("Blink Detection with Gaze Tracking")

blink_count = 0
blink_text = ""
start_time = time.time()
warning_time = 3  # Set the warning time to 3 seconds

if st.checkbox("Show webcam feed"):
    stframe = st.empty()
    warning_placeholder = st.empty()  # Create a placeholder for the warning message

    while True:
        ret, frame = webcam.read()
        if not ret:
            st.error("Failed to read frame from webcam")
            break

        text = ""

        if frame is not None:
            # Preprocess the input frame
            preprocessed_frame = preprocess(frame)

            # Pass the preprocessed frame through the model
            encoder_output = encoder_exec.infer(inputs={encoder_net.input_info.keys()[0]: preprocessed_frame})
            decoder_output = decoder_exec.infer(inputs={decoder_net.input_info.keys()[0]: encoder_output})
            driver_action = postprocess(decoder_output)

            if driver_action == "blink":
                blink_count += 1
                if blink_count == 1:
                    start_time = time.time()  # Start the timer when the first blink is detected
                elif blink_count >= warning_time:
                    blink_text = "Blink Detected for 3 seconds or more!"

            else:
                blink_count = 0
                blink_text = ""
                start_time = time.time()  # Reset the timer when no blink is detected
                warning_placeholder.empty()  # Hide the warning message

            if time.time() - start_time >= warning_time and blink_count > 0:
                blink_text = "Blink Detected for 3 seconds or more!"
                warning_placeholder.warning(blink_text)  # Show the warning message once the timer reaches 3 seconds

            if gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil: " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) == 27:
            break

webcam.release()
cv2.destroyAllWindows()


