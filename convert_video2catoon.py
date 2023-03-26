import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Get FPS and calculate the waiting time in millisecond
fps = cap.get(cv.CAP_PROP_FPS)
wait_msec = int(1 / fps * 1000)

while True:
    # Read an image from 'video'
    ret, video = cap.read()
    if not ret:
        break

    video_gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    # Show the image
    cv.imshow('Video Player', video_gray)

    # Terminate if the given key is ESC
    key = cv.waitKey(wait_msec)
    if key == 27: # ESC
        break

cv.destroyAllWindows()