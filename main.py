from gender_detection import gender_recognition
video_path = 'VIDEO_PATH'

try:
  gender_recognition(video_path)
except Exception as e:
  print(f'something went wrong {e}')

