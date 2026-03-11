import h5py
import cv2

H5_PATH = r"C:\Users\pardo\Desktop\3DV\simulation\data\20250804_104715.h5"

# Play the video
with h5py.File(H5_PATH, "r") as f:
    images = f["observations/images/aria_rgb_cam/color"]

    for i in range(images.shape[0]):
        frame = images[i]

        # convert RGB -> BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Aria RGB Cam", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # press ESC to stop
            break

cv2.destroyAllWindows()