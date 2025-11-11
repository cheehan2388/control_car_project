import cv2
import numpy as np
import serial
import time
import skfuzzy as fuzz
from skfuzzy import control as fuzz_control

 
 
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE    = 115200
BASE_SPEED   = 30      # 巡線時的基準速度
TURN_SPEED_ONE   = 70 #右轉 (三角)
TURN_SPEED_TWO  = 50# 右轉 
NO_ARROW_MAX_First = 5       # 右轉模式下連續 N 幀都沒看到箭頭就恢復巡線
NO_ARROW_MAX_2 =11
MIN_AREA     = 90500    # 濾除雜訊的小面積三角形
MAX_AREA     = 110000
CAM_WIDTH        = 1280          #寬
OFFSET_RANGE     = np.arange(-CAM_WIDTH//2, CAM_WIDTH//2 + 1, 1)
OFFSET_RATE_RANGE= np.arange(-100, 101, 1)
SPEED_DIFF_RANGE = np.arange(-60,  61,  1)

ALPHA_SMOOTH =  0.5

count = 0

# 初始化硬體

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera at index 0")
    exit(1)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection failed: {e}")
    exit(1)



offset_rate = fuzz_control.Antecedent(OFFSET_RATE_RANGE,'offset_rate') 
offset      = fuzz_control.Antecedent(OFFSET_RANGE,    'offset')#條
speed_diff  = fuzz_control.Consequent(SPEED_DIFF_RANGE,'speed_diff') #結果



# 5 個模糊集合 
offset['far_left']  = fuzz.trimf(OFFSET_RANGE, [-CAM_WIDTH//2, -CAM_WIDTH//2, -CAM_WIDTH//4])
offset['left']      = fuzz.trimf(OFFSET_RANGE, [-CAM_WIDTH//2, -CAM_WIDTH//4, -CAM_WIDTH//8])
offset['center']    = fuzz.trimf(OFFSET_RANGE, [-CAM_WIDTH//5,             0,  CAM_WIDTH//5])
offset['right']     = fuzz.trimf(OFFSET_RANGE, [ CAM_WIDTH//8,  CAM_WIDTH//4,  CAM_WIDTH//2])
offset['far_right'] = fuzz.trimf(OFFSET_RANGE, [ CAM_WIDTH//4,  CAM_WIDTH//2,  CAM_WIDTH//2])

offset_rate['neg_large'] = fuzz.trimf(OFFSET_RATE_RANGE, [-100, -100, -40])
offset_rate['neg_small'] = fuzz.trimf(OFFSET_RATE_RANGE, [ -80,  -40,   0])
offset_rate['zero']      = fuzz.trimf(OFFSET_RATE_RANGE, [  -5,    0,   5])
offset_rate['pos_small'] = fuzz.trimf(OFFSET_RATE_RANGE, [   0,   40,  80])
offset_rate['pos_large'] = fuzz.trimf(OFFSET_RATE_RANGE, [  40,  100, 100])

speed_diff['sharp_left']   = fuzz.trimf(SPEED_DIFF_RANGE, [-80, -80, -30])
speed_diff['slight_left']  = fuzz.trimf(SPEED_DIFF_RANGE, [-35, -20,   0])
speed_diff['straight']     = fuzz.trimf(SPEED_DIFF_RANGE, [ -5,   0,   5])
speed_diff['slight_right'] = fuzz.trimf(SPEED_DIFF_RANGE, [   0,  20,  35])
speed_diff['sharp_right']  = fuzz.trimf(SPEED_DIFF_RANGE, [ 30,  80,  80])

#previo
# speed_diff['sharp_left']   = fuzz.trimf(SPEED_DIFF_RANGE, [-60, -60, -30])
# speed_diff['slight_left']  = fuzz.trimf(SPEED_DIFF_RANGE, [-35, -20,   0])
# speed_diff['straight']     = fuzz.trimf(SPEED_DIFF_RANGE, [ -5,   0,   5])
# speed_diff['slight_right'] = fuzz.trimf(SPEED_DIFF_RANGE, [   0,  20,  35])
# speed_diff['sharp_right']  = fuzz.trimf(SPEED_DIFF_RANGE, [ 30,  60,  60])

# rules = [
#     fuzz_control.Rule(offset['far_left']  | offset_rate['neg_large'], speed_diff['sharp_left']),
#     fuzz_control.Rule(offset['left']      | offset_rate['neg_small'], speed_diff['slight_left']),
#     fuzz_control.Rule(offset['center']    & offset_rate['zero'],      speed_diff['straight']),
#     fuzz_control.Rule(offset['right']     | offset_rate['pos_small'], speed_diff['slight_right']),
#     fuzz_control.Rule(offset['far_right'] | offset_rate['pos_large'], speed_diff['sharp_right']),
# ]
rules2 = [
    fuzz_control.Rule(offset['far_left']   , speed_diff['sharp_left']),
    fuzz_control.Rule(offset['left']       , speed_diff['slight_left']),
    fuzz_control.Rule(offset['center']     ,      speed_diff['straight']),
    fuzz_control.Rule(offset['right']      , speed_diff['slight_right']),
    fuzz_control.Rule(offset['far_right']  , speed_diff['sharp_right']),
]
control_system = fuzz_control.ControlSystem(rules2)
controller     = fuzz_control.ControlSystemSimulation(control_system)

# 工具函式

def region_int(img):
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0*w), h),
        (int(0*w), int(0.75*h)),
        (int(w), int(0.75*h)),
        (int(w), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)

def region_int_t(img):
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.03*w), h),
        (int(0.03*w), int(0*h)),
        (int(0.97*w), int(0*h)),
        (int(.97*w), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask) 

def send_pwm(left_pwm, right_pwm):
    left_pwm  = max(min(int(left_pwm),  255), -255)
    right_pwm = max(min(int(right_pwm), 255), -255)
    cmd = f'L{left_pwm}R{right_pwm}\n'
    try:
        ser.write(cmd.encode())
        ser.flush()
        time.sleep(0.02)
    except serial.SerialException as e:
        print(f"Serial write error: {e}")

# def calculate_offset(lines, frame_width):
#     if lines is None or len(lines) == 0:
#       return 0
#     offsets = [(x1 + x2) / 2 - frame_width / 2 for x1, _, x2, _ in lines[:,0]]
#     return np.mean(offsets)

def calculate_offset(lines, frame_width):

    if lines is None or len(lines) == 0:
        return 0

    center_x = frame_width / 2
    left_pts, right_pts = [], []

    # 分組並收集中點
    for x1, y1, x2, y2 in lines[:,0]:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if mx < center_x:
            left_pts .append((mx, my))
        else:
            right_pts.append((mx, my))

    # 決定 left區, right
    if left_pts and right_pts:
        # 取各自最靠近底部(最大my)的那條
        left_x  = max(left_pts,  key=lambda p: p[1])[0]
        right_x = max(right_pts, key=lambda p: p[1])[0]
    elif left_pts:
        left_x  = max(left_pts, key=lambda p: p[1])[0]
        right_x = frame_width
    elif right_pts:
        left_x  = 0
        right_x = max(right_pts, key=lambda p: p[1])[0]
    else:
        return 0

    # 算中心偏移
    lane_center = (left_x + right_x) / 2
    return lane_center - center_x


def arrow_direction(img, max_area = MAX_AREA,min_area=MIN_AREA):

    roi  = region_int_t(img)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Otsu 二值化並反相，讓箭頭變白 （嘗試過不做反相差別很大
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for cnt in contours:
        # 噪音處理
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:           
            continue

        
        peri   = cv2.arcLength(cnt, True)                     
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)     
        if len(approx) != 3:
            continue

      
        pts = approx.reshape((3, 2)) 

        #算他們的角度選最大然後最大如果在右邊就是右箭頭
        def angle_at(p, q, r):
             
            v1 = p - q
            v2 = r - q
            cosθ = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cosθ, -1, 1))
            
        centroid = np.mean(pts, axis=0).astype(int)

        angles = [
            angle_at(pts[(i+1)%3], pts[i], pts[(i+2)%3])
            for i in range(3)
        ]
        tip_idx = int(np.argmin(angles))       
        tip      = pts[tip_idx]                
        base_pts = np.delete(pts, tip_idx, 0)  
        base_ctr = base_pts.mean(axis=0)       

        
        direction = tip - base_ctr            # [dx, dy]
        
         
        cv2.drawContours(img, [approx], 0, (255,0,0), 2)


         
        area_text = f"{int(area)} px^2"
        cv2.putText(img,
                    area_text,
                    tuple(centroid + np.array([0, 10])),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3, (255,0,0), 2)


        
    

        return 'right' if direction[0] > 0 else 'left' 

# 主迴圈：巡線 + 箭頭右轉 

turning_right = False
turning_left  = False
prev_offset       = 0.0    
no_right_arrow_cnt = 0
no_left_arrow_cnt  = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

   
       #先做線檢測 → 計算 offset_smooth, offset_rate
    
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5,5), 0)
        edges  = cv2.Canny(blur, 50, 150)
        masked = region_int(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        lines  = cv2.HoughLinesP(
            cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY),
            1, np.pi/180, 18,
            minLineLength=35, maxLineGap=10
        )

        # 原始偏移量
        offset_raw      = calculate_offset(lines, frame.shape[1])
        # EMA 平滑
        offset_smooth   = ALPHA_SMOOTH * offset_raw + (1 - ALPHA_SMOOTH) * prev_offset
   
        offset_rate_val = offset_smooth - prev_offset
        prev_offset     = offset_smooth

        
        # 2) 箭頭偵測 → 狀態切換
        
        dirn = arrow_direction(frame)   

        # 右轉 
        if turning_right:
            if dirn == 'right':
                no_right_arrow_cnt = 0
            else:
                no_right_arrow_cnt += 1
                if count == 0 :
                    if no_right_arrow_cnt >= NO_ARROW_MAX_First:
                        turning_right      = False
                        no_right_arrow_cnt = 0
                        count +=1
                else:
                    if no_right_arrow_cnt >= NO_ARROW_MAX_2:
                        turning_right      = False
                        no_right_arrow_cnt = 0
                        
                    

        # 左轉 
        elif turning_left:
            if dirn == 'left':
                no_left_arrow_cnt = 0
            else:
                no_left_arrow_cnt += 1
                if no_left_arrow_cnt >= NO_ARROW_MAX_First:
                    turning_left      = False   # ← 修正：關掉 left，非 right
                    no_left_arrow_cnt = 0

        
        else:
            if dirn == 'right':
                turning_right      = True
                no_right_arrow_cnt = 0
            elif dirn == 'left':
                turning_left      = True
                no_left_arrow_cnt = 0

      
        #   根據狀態決定左右輪速度
         
        if turning_right:
           
            left_speed, right_speed = TURN_SPEED_ONE, -TURN_SPEED_TWO
            
        elif turning_left:
      
            left_speed, right_speed = -TURN_SPEED_TWO, TURN_SPEED_ONE
        else:
            # LINE FOLLOW 模糊推論
            try:
                controller.input['offset']      = offset_smooth
                
                controller.compute()
                diff = controller.output['speed_diff']
            except:
                diff = 0
            left_speed  = BASE_SPEED - diff/2
            right_speed = BASE_SPEED + diff/2

  
    
        
        send_pwm(left_speed, right_speed)

        mode = "TURN R" if turning_right else ("TURN L" if turning_left else "LINE")
        cv2.putText(frame, mode, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, f"Offset:{offset_smooth:+.1f}  dOff:{offset_rate_val:+.1f}", 
                    (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if dirn:
            cv2.putText(frame, f"ARROW→{dirn.upper()}", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if not turning_right:
            if lines is not None:
                for x1,y1,x2,y2 in lines[:,0]:
                    cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            h, w = frame.shape[:2]
            poly = np.array([[
       (int(0*w), h),
        (int(0*w), int(0.49*h)),
        (int(w), int(0.50*h)),
        (int(w), h)
            ]], dtype=np.int32)
            cv2.polylines(frame, poly, True, (255,255,0), 2)

        cv2.putText(frame, f"L:{int(left_speed)} R:{int(right_speed)}",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        if dirn == 'right':
            cv2.putText(frame, "ARROW→", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        elif dirn == 'left':
             cv2.putText(frame, "ARROW<-", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.imshow('View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    send_pwm(0,0)
    cap.release()
    ser.close()
    cv2.destroyAllWindows()
    print("Cleanup complete")



