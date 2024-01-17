import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('bottles4.mp4')

# Define color range for the black conveyor in HSV color space
lower_black = np.array([0, 0, 0])
upper_black = np.array([0,0,0])

# Initialize bottle count
bottle_count = 0

# Initialize frame counter
frame_counter = 0

# Define how many frames a bottle stays in view
frames_per_bottle = 60

# Define size threshold for the contours, change this to adjust the contour size
size_threshold = 5000

while True:
    # Read the next frame
    ret, frame = cap.read()

    # If the frame was successfully read
    if ret:
        # Increase the frame counter
        frame_counter += 1

        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Generate mask for the color of the conveyor
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Invert the mask to get the bottles
        mask = cv2.bitwise_not(mask)

        # Perform morphological operations to remove small blobs
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > size_threshold]

        # If it's time to count the bottles, do it
        if frame_counter % frames_per_bottle == 0:
            for cnt in contours:
                bottle_count += 1
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw the bottle count at the center of the contour
                cv2.putText(frame, f"{bottle_count}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw all contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Display the current total bottle count
        cv2.putText(frame, f"Total Count: {bottle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with contours
        cv2.imshow("Contours", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('End of video reached. Press any key to exit...')
        cv2.waitKey(0)
        break



cap.release()
cv2.destroyAllWindows()
