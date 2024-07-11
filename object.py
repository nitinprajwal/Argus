import argparse
import sys
import time
import pyttsx3

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Global variables for cooldown
COOLDOWN_DURATION = 3  # Cooldown duration in seconds
LAST_AUDIO_TIME = 0  # Time of the last audio output


def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  detection_frame = None
  detection_result_list = []

  
  def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    global FPS, COUNTER, START_TIME, LAST_AUDIO_TIME

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    detection_result_list.append(result)
    COUNTER += 1

    current_time = time.time()

    # Provide audio output for the detected objects with score > 0.55
    for detection in result.detections:
        label = detection.categories[0].category_name
        # Calculate the area of the bounding box
        bbox = detection.bounding_box
        bbox_area = bbox.width * bbox.height
        # Find the detection with the largest bounding box area
        max_area_detection = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        max_area_label = max_area_detection.categories[0].category_name
        max_area_score = max_area_detection.categories[0].score

        # Get the bounding box coordinates of the maximum area detection
        max_bbox = max_area_detection.bounding_box
        max_x_min, max_y_min, max_x_max, max_y_max = max_bbox.origin_x, max_bbox.origin_y, max_bbox.origin_x + max_bbox.width, max_bbox.origin_y + max_bbox.height

        # Calculate the center position of the maximum area bounding box
        max_center_x = (max_x_min + max_x_max) / 2
        max_center_y = (max_y_min + max_y_max) / 2

        # Normalize the center position to a value between 0 and 1
        frame_width, frame_height = 640, 480  # Update these values if you change the frame size
        normalized_max_center_x = max_center_x / frame_width
        normalized_max_center_y = max_center_y / frame_height

        # Determine the audio instruction based on the position of the maximum area object
        if normalized_max_center_x < 0.3:
            position_instruction = "Object is on the left side."
        elif normalized_max_center_x > 0.7:
            position_instruction = "Object is on the right side."
        else:
            position_instruction = "Object is in the center."

        if normalized_max_center_y < 0.3:
            position_instruction += " It is near the top."
        elif normalized_max_center_y > 0.7:
            position_instruction += " It is near the bottom."
        else:
            position_instruction += " It is in the middle."

        # Provide audio output for the maximum area detection
        if max_area_score >= 0.4 and current_time - LAST_AUDIO_TIME >= COOLDOWN_DURATION:
            engine.say(f"Detected: {max_area_label}. {position_instruction}")
            engine.runAndWait()
            LAST_AUDIO_TIME = current_time

  # Initialize the object detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.ObjectDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                         max_results=max_results, score_threshold=score_threshold,
                                         result_callback=save_result)
  detector = vision.ObjectDetector.create_from_options(options)
  engine = pyttsx3.init()


  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    image=cv2.resize(image,(640,480))
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    #image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run object detection using the model.
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if detection_result_list:
        # print(detection_result_list)
        current_frame = visualize(current_frame, detection_result_list[0])
        detection_frame = current_frame
        detection_result_list.clear()

    if detection_frame is not None:
        cv2.imshow('object_detection', detection_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break

  detector.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
#      default='best.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max number of detection results.',
      required=False,
      default=5)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of detection results.',
      required=False,
      type=float,
      default=0.25)
  # Finding the camera ID can be very reliant on platform-dependent methods. 
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults),
      args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()