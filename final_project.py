import cv2
import numpy as np
import serial
import time
import skfuzzy as fuzz
from skfuzzy import control as fuzz_control

# ==============================
# 參數設定
# ==============================
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE    = 115200
BASE_SPEED   = 20      # 巡線時的基準速度
TURN_SPEED   = 50      # 右轉時左輪正轉 PWM（右輪反轉）
NO_ARROW_MAX = 5       # 右轉模式下連續 N 幀都沒看到箭頭就恢復巡線
MIN_AREA     = 6000     # 濾除雜訊的小面積三角形
CAM_WIDTH        = 1280          # ← 若後面有改解析度，這裡跟著改
OFFSET_RANGE     = np.arange(-CAM_WIDTH//2, CAM_WIDTH//2 + 1, 1)
OFFSET_RATE_RANGE= np.arange(-100, 101, 1)
SPEED_DIFF_RANGE = np.arange(-60,  61,  1)

ALPHA_SMOOTH =  1
# ==============================
# 初始化硬體
# ==============================
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

# ==============================
# 模糊控制器設定（原樣照搬）



offset_rate = fuzz_control.Antecedent(OFFSET_RATE_RANGE,'offset_rate') 
offset      = fuzz_control.Antecedent(OFFSET_RANGE,    'offset')#條
speed_diff  = fuzz_control.Consequent(SPEED_DIFF_RANGE,'speed_diff') #結果



# -- 5 個模糊集合 --
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

speed_diff['sharp_left']   = fuzz.trimf(SPEED_DIFF_RANGE, [-60, -60, -30])
speed_diff['slight_left']  = fuzz.trimf(SPEED_DIFF_RANGE, [-45, -20,   0])
speed_diff['straight']     = fuzz.trimf(SPEED_DIFF_RANGE, [ -5,   0,   5])
speed_diff['slight_right'] = fuzz.trimf(SPEED_DIFF_RANGE, [   0,  20,  45])
speed_diff['sharp_right']  = fuzz.trimf(SPEED_DIFF_RANGE, [ 30,  60,  60])

rules = [
    fuzz_control.Rule(offset['far_left']  | offset_rate['neg_large'], speed_diff['sharp_left']),
    fuzz_control.Rule(offset['left']      | offset_rate['neg_small'], speed_diff['slight_left']),
    fuzz_control.Rule(offset['center']    & offset_rate['zero'],      speed_diff['straight']),
    fuzz_control.Rule(offset['right']     | offset_rate['pos_small'], speed_diff['slight_right']),
    fuzz_control.Rule(offset['far_right'] | offset_rate['pos_large'], speed_diff['sharp_right']),
]
control_system = fuzz_control.ControlSystem(rules)
controller     = fuzz_control.ControlSystemSimulation(control_system)
# ==============================
# 工具函式
# ==============================
def region_int(img):
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.15*w), h),
        (int(0.15*w), int(0.49*h)),
        (int(0.86*w), int(0.50*h)),
        (int(0.86*w), h)
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

def calculate_offset(lines, frame_width):
    # if lines is None or len(lines) == 0:
    #     return -320
    offsets = [(x1 + x2) / 2 - frame_width / 2 for x1, _, x2, _ in lines[:,0]]
    return np.mean(offsets)

def arrow_direction(img, min_area=MIN_AREA):
    """
    偵測是否有黑底三角箭頭，回傳 'left' / 'right' / None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu 二值化並反相，讓箭頭變白
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 選最大輪廓
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None

    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    if len(approx) != 3:
        return None

    pts      = approx.reshape(-1, 2)
    centroid = pts.mean(axis=0)
    # 找最尖點：取與重心距離最大的頂點
    dists    = np.linalg.norm(pts - centroid, axis=1)
    tip_idx  = np.argmax(dists)
    tip      = pts[tip_idx]

    return 'right' if tip[0] > centroid[0] else 'left'

# ==============================
# 主迴圈：巡線 + 箭頭右轉模式
# ==============================
turning_right = False
turning_left  = False
prev_offset       = 0.0   # EMA 初始值
no_right_arrow_cnt = 0
no_left_arrow_cnt  = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----------------------------------------
        # 1) 先做線檢測 → 計算 offset_smooth, offset_rate
        # ----------------------------------------
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
        # 變化率（D 項）
        offset_rate_val = offset_smooth - prev_offset
        prev_offset     = offset_smooth

        # ----------------------------------------
        # 2) 箭頭偵測 → 狀態切換
        # ----------------------------------------
        dirn = arrow_direction(frame)  # 'left' / 'right' / None

        # 右轉中
        if turning_right:
            if dirn == 'right':
                no_right_arrow_cnt = 0
            else:
                no_right_arrow_cnt += 1
                if no_right_arrow_cnt >= NO_ARROW_MAX:
                    turning_right      = False
                    no_right_arrow_cnt = 0

        # 左轉中
        elif turning_left:
            if dirn == 'left':
                no_left_arrow_cnt = 0
            else:
                no_left_arrow_cnt += 1
                if no_left_arrow_cnt >= NO_ARROW_MAX:
                    turning_left      = False   # ← 修正：關掉 left，非 right
                    no_left_arrow_cnt = 0

        # 都不轉時，看新箭頭
        else:
            if dirn == 'right':
                turning_right      = True
                no_right_arrow_cnt = 0
            elif dirn == 'left':
                turning_left      = True
                no_left_arrow_cnt = 0

        # ----------------------------------------
        # 3) 根據狀態決定左右輪速度
        # ----------------------------------------
        if turning_right:
            left_speed, right_speed = TURN_SPEED, -TURN_SPEED
        elif turning_left:
            left_speed, right_speed = -TURN_SPEED, TURN_SPEED
        else:
            # LINE FOLLOW 模糊推論
            try:
                controller.input['offset']      = offset_smooth
                controller.input['offset_rate'] = offset_rate_val
                controller.compute()
                diff = controller.output['speed_diff']
            except:
                diff = 0
            left_speed  = BASE_SPEED - diff/2
            right_speed = BASE_SPEED + diff/2

        # ----------------------------------------
        # 4) 傳送 PWM & 顯示
        # ----------------------------------------
        send_pwm(left_speed, right_speed)

        mode = "TURN R" if turning_right else ("TURN L" if turning_left else "LINE")
        cv2.putText(frame, mode, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, f"Offset:{offset_smooth:+.1f}  dOff:{offset_rate_val:+.1f}", 
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        if dirn:
            cv2.putText(frame, f"ARROW→{dirn.upper()}", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if not turning_right:
            if lines is not None:
                for x1,y1,x2,y2 in lines[:,0]:
                    cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            h, w = frame.shape[:2]
            poly = np.array([[
       (int(0.15*w), h),
        (int(0.15*w), int(0.49*h)),
        (int(0.86*w), int(0.50*h)),
        (int(0.86*w), h)
            ]], dtype=np.int32)
            cv2.polylines(frame, poly, True, (255,255,0), 2)

        mode_text = "TURN RIGHT" if turning_right else "LINE FOLLOW"
        cv2.putText(frame, mode_text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, f"L:{int(left_speed)} R:{int(right_speed)}",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        if dirn == 'right':
            cv2.putText(frame, "ARROW→", (10,110),
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
