import pyzed.sl as sl
import cv2
import numpy as np
import math

def do_nothing(x):
    pass

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit()
    runtime_parameters = sl.RuntimeParameters()
    
    image_zed = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    cv2.namedWindow("Trackbars")
    
    cv2.createTrackbar("Low Hue", "Trackbars", 0, 179, do_nothing)
    cv2.createTrackbar("Low Saturation", "Trackbars", 0, 255, do_nothing)
    cv2.createTrackbar("Low Value", "Trackbars", 0, 255, do_nothing)
    cv2.createTrackbar("High Hue", "Trackbars", 179, 179, do_nothing)
    cv2.createTrackbar("High Saturation", "Trackbars", 255, 255, do_nothing)
    cv2.createTrackbar("High Value", "Trackbars", 255, 255, do_nothing)

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("Low Hue", "Trackbars")
            ls = cv2.getTrackbarPos("Low Saturation", "Trackbars")
            lv = cv2.getTrackbarPos("Low Value", "Trackbars")
            hh = cv2.getTrackbarPos("High Hue", "Trackbars")
            hs = cv2.getTrackbarPos("High Saturation", "Trackbars")
            hv = cv2.getTrackbarPos("High Value", "Trackbars")


            lower_bound = np.array([lh, ls, lv])
            upper_bound = np.array([hh, hs, hv])


            mask = cv2.inRange(hsv, lower_bound, upper_bound)


            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            num_labels, labels, stats, centroids = output

            for i in range(1, num_labels):

                x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
                if area > 150:  
                   
                    center_x, center_y = int(centroids[i][0]), int(centroids[i][1])


                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                    err, point_cloud_value = point_cloud.get_value(center_x, center_y)


                    if math.isfinite(point_cloud_value[2]):
                        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                             point_cloud_value[1] * point_cloud_value[1] +
                                             point_cloud_value[2] * point_cloud_value[2])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if center_x < 640:
                        cv2.putText(frame, f"LEFT Distance: {distance:.2f} mm, OFFSET: {point_cloud_value[0]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, f" RIGHT Distance: {distance:.2f} mm, OFFSET: {point_cloud_value[0]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    


            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

