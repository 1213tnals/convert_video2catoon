import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

# Get FPS and calculate the waiting time in millisecond
fps = video.get(cv.CAP_PROP_FPS)
wait_msec = int(1 / fps * 1000)

while True:
    # Read an image from 'video'
    valid, img = video.read()
    if not valid:
        break

    # Show the image
    cv.imshow('Video Player', img)

    # Terminate if the given key is ESC
    key = cv.waitKey(wait_msec)
    if key == 27: # ESC
        break

cv.destroyAllWindows()