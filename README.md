# control_car_project （控制實驗 期終專案）
final_project.py : 

i. Use fuzzy control with two parameters:
    • the offset between the black line and the center of the camera
    • the difference between the previous offset and the latest offset

ii. Implement a Hough Transform and Canny to detect the line from the camera.

iii. Detect triangles and extract their contours. Measure their angles, compare them, and select the largest angle. If the largest angle appears on the left, then turn right, and vice versa. The turning duration is determined by a counter and the triangular area size.
    
iv. arduino programming for motor and raspberry pie for embedded system
