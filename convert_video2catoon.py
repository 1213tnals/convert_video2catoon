import cv2 as cv
import numpy as np
from Sobel_edge import drawText

# get webcam data
cap = cv.VideoCapture(0)

img_threshold_type = cv.THRESH_BINARY_INV

# Initialize control parameters
threshold = 127
adaptive_type = cv.ADAPTIVE_THRESH_MEAN_C
adaptive_blocksize = 99
adaptive_C = 4
# Initialize control parameters
threshold1 = 500
threshold2 = 1200
aperture_size = 5
img_select = -1
alpha = 0.5

# Get FPS and calculate the waiting time in millisecond
fps = cap.get(cv.CAP_PROP_FPS)
wait_msec = int(1 / fps * 1000)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    # Read an image from 'video'
    ret, video = cap.read()
    if not ret:
        break

    # Convert video to video_gray
    video_gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    # Get the Canny edge image
    edge = cv.Canny(video_gray, threshold1, threshold2, apertureSize=aperture_size)

    # Apply thresholding to the image
    # _, binary_user = cv.threshold(video_gray, threshold, 255, img_threshold_type)
    # threshold_otsu, binary_otsu = cv.threshold(video_gray, threshold, 255, img_threshold_type | cv.THRESH_OTSU)
    binary_adaptive = cv.adaptiveThreshold(video_gray, 255, adaptive_type, img_threshold_type, adaptive_blocksize, adaptive_C)

    # Show the image and its thresholded result
    # drawText(binary_user, f'Threshold: {threshold}')
    # drawText(binary_otsu, f'Otsu Threshold: {threshold_otsu}')
    adaptive_type_text = 'M' if adaptive_type == cv.ADAPTIVE_THRESH_MEAN_C else 'G'
    drawText(binary_adaptive, f'Type: {adaptive_type_text}, BlockSize: {adaptive_blocksize}, C: {adaptive_C}')

    # Convert binary data to color data step1
    binary_adaptive = cv.cvtColor(binary_adaptive, cv.COLOR_GRAY2BGR)

    # Convert binary data to color data step2
    binary_adaptive = cv.resize(binary_adaptive, (video.shape[1], video.shape[0]))

    # Apply alpha blending (you should make 2 data to color (type) data)
    blend = (alpha * binary_adaptive + (1 - alpha) * video).astype(np.uint8) # Alternative) cv.addWeighted()

    # Show the image
    cv.imshow('Video Player', blend)

    # Terminate if the given key is ESC
    key = cv.waitKey(wait_msec)
    if key == 27: # ESC
        break
    elif key == ord('+') or key == ord('='):
        alpha = min(alpha + 0.1, 1)
    elif key == ord('-') or key == ord('_'):
        alpha = max(alpha - 0.1, 0)

# release cap object and close all windows
cap.release()
cv.destroyAllWindows()