import cv2 as cv
import numpy as np
from Sobel_edge import drawText

# get webcam data
cap = cv.VideoCapture(0)

img_threshold_type = cv.THRESH_BINARY_INV

# Initialize control parameters
adaptive_type = cv.ADAPTIVE_THRESH_MEAN_C
adaptive_blocksize = 99
adaptive_C = 4
threshold1 = 500
threshold2 = 1500
aperture_size = 5
img_select = -1
alpha = 0.4
n_iterations = 1
mode = 0

# Get FPS and calculate the waiting time in millisecond
fps = cap.get(cv.CAP_PROP_FPS)
wait_msec = int(1 / fps * 1000)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Read an image from 'video'
    ret, video = cap.read()
    if not ret:
        break

    # Blurring Image
    video = cv.medianBlur(video, 5)

    # Convert video to video_gray
    video_gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    # Apply thresholding to the image
    binary_adaptive = cv.adaptiveThreshold(video_gray, 255, adaptive_type, img_threshold_type, adaptive_blocksize, adaptive_C)

    # Get the Canny edge image
    if(mode == 0):
        video_edge = cv.Canny(video_gray, threshold1, threshold2, apertureSize=aperture_size)
        video_changed = cv.morphologyEx(video_edge, cv.MORPH_DILATE, np.ones((2, 2), dtype=np.uint8), iterations=n_iterations)
    elif(mode == 1):
        video_edge = cv.Canny(binary_adaptive, threshold1, threshold2, apertureSize=aperture_size)
        video_changed = cv.morphologyEx(video_edge, cv.MORPH_DILATE, np.ones((2, 2), dtype=np.uint8), iterations=n_iterations)
    elif(mode == 2):
        video_edge = cv.adaptiveThreshold(video_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
        video_changed = video_edge

    # Convert binary data to color data step1
    video_changed = cv.cvtColor(video_changed, cv.COLOR_GRAY2BGR)

    # Convert binary data to color data step2
    video_changed = cv.resize(video_changed, (video.shape[1], video.shape[0]))

    # Apply alpha blending (you should make 2 data to color (type) data)
    blend = (alpha * video_changed + (1 - alpha) * video).astype(np.uint8) # Alternative) cv.addWeighted()

    # Show the image
    cv.imshow('Cartoon Player', blend)

    # Terminate if the given key is ESC
    key = cv.waitKey(wait_msec)
    if key == 27: # ESC
        break
    elif key == ord('+') or key == ord('='):
        threshold1 += 2
    elif key == ord('-') or key == ord('_'):
        threshold1 -= 2
    elif key == ord(']') or key == ord('}'):
        threshold2 += 2
    elif key == ord('[') or key == ord('{'):
        threshold2 -= 2
    elif key == ord('>') or key == ord('.'):
        aperture_size = min(aperture_size + 2, 7)
    elif key == ord('<') or key == ord(','):
        aperture_size = max(aperture_size - 2, 3)
    elif key == ord('a'):
        mode = 0
    elif key == ord('b'):
        mode = 1
    elif key == ord('c'):
        mode = 2
    
# release cap object and close all windows
cap.release()
cv.destroyAllWindows()