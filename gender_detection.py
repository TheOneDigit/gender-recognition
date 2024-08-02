import cv2
from deepface import DeepFace
from tqdm.notebook import tqdm

def gender_recognition(video_path: str):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error opening video file")
      exit()

  # Get video properties
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
  fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Create VideoWriter object to save the output video
  out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames


  # Load face cascade classifier
  facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  pbar = tqdm(total=total_frames)

  while True:
      ret, frame = cap.read()

      if not ret:
          break  # Exit loop when video ends or frame read error

      try:
        result = DeepFace.analyze(frame, actions=['gender'], detector_backend='yolov8')
      except:
        continue

      # Convert frame to grayscale and detect faces
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

      # Draw rectangles around detected faces and add emotion text above the bounding box
      font = cv2.FONT_HERSHEY_SIMPLEX
      for (x, y, w, h) in faces:
          # Draw bounding box
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)

          # Add text above bounding box
          if result:
              most_confident_result = max(result, key=lambda x: x['face_confidence'])
              gender = most_confident_result['dominant_gender']
              text_size, _ = cv2.getTextSize(gender, font, 1, 2)
              text_x = x
              text_y = y - 10 if y - 10 > 10 else y + 10  # Ensure text doesn't go off the top of the frame

              font = cv2.FONT_HERSHEY_SIMPLEX
              fontScale = 150/h  # Increase for larger text
              color = (0, 0, 255)  # Green color
              thickness = 2  # Increase for bolder text
              cv2.putText(frame, gender, (text_x, text_y), font, fontScale, color, thickness, cv2.LINE_AA)
      # Write the processed frame to output video
      out.write(frame)
      pbar.update(1)


  # Release resources
  cap.release()
  out.release()
  cv2.destroyAllWindows()

  return 'Video Process Successfully with name output_video.avi'
