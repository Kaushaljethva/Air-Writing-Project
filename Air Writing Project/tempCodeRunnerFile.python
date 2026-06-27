import cv2
import numpy as np
import mediapipe as mp

# MediaPipe હેન્ડ ટ્રેકિંગ સેટઅપ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# હાથના લીલા પોઇન્ટ્સ
hand_dot_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
hand_line_spec = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)

# 💻 લિંક્ડઇન સુપર પ્રોફેશનલ આઇટી કલર્સ (ગોલ્ડન હટાવીને VIOLET ઉમેર્યો)
colors = [
    (80, 80, 240),    # 0: Coral Rose (સોફ્ટ પ્રીમિયમ રેડ/પિંક)
    (235, 130, 0),    # 1: Cobalt Blue (એકદમ પ્રોફેશનલ આઇટી બ્લુ)
    (140, 220, 0),    # 2: Frosted Mint (શાંત અને સુંદર ગ્રીન)
    (211, 0, 148)     # 3: Deep Electric Violet / Indigo (એકદમ ઘાટો અને ક્લાસિક જાંબલી)
]
color_names = ["ROSE", "BLUE", "MINT", "VIOLET"]
color_index = 0

canvas = None
xp, yp = 0, 0
color_timer = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- 🔄 ઓટોમેટિક કલર ચેન્જ લોજિક ---
    if results.multi_hand_landmarks and color_index != 5:
        color_timer += 1
        if color_timer > 50:
            color_index = (color_index + 1) % 4
            color_timer = 0

    current_color = (0, 0, 0) if color_index == 5 else colors[color_index]

    # --- 🎛️ વ્હાઇટ આઉટલાઇન વાળું સાઇડ પેનલ ---
    cv2.rectangle(frame, (w - 110, 15), (w - 10, h - 15), (10, 10, 10), cv2.FILLED)
    cv2.rectangle(frame, (w - 110, 15), (w - 10, h - 15), (255, 255, 255), 1)
    
    # ૨x૨ ગ્રીડ પોઝિશન્સ
    positions = [
        (w - 80, 50),   # Row 1, Left  -> Rose
        (w - 40, 50),   # Row 1, Right -> Blue
        (w - 80, 150),  # Row 2, Left  -> Mint
        (w - 40, 150)   # Row 2, Right -> Violet
    ]
    
    # કલર સર્કલ્સ
    cv2.circle(frame, positions[0], 12, colors[0], cv2.FILLED)
    cv2.circle(frame, positions[1], 12, colors[1], cv2.FILLED)
    cv2.circle(frame, positions[2], 12, colors[2], cv2.FILLED)
    cv2.circle(frame, positions[3], 12, colors[3], cv2.FILLED)
    
    # કલરના નામ નાના વ્હાઇટ અક્ષરોમાં
    cv2.putText(frame, color_names[0], (positions[0][0] - 12, positions[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(frame, color_names[1], (positions[1][0] - 12, positions[1][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(frame, color_names[2], (positions[2][0] - 12, positions[2][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(frame, color_names[3], (positions[3][0] - 20, positions[3][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # CLEAR બટન
    cv2.rectangle(frame, (w - 95, h - 55), (w - 25, h - 25), (30, 30, 30), cv2.FILLED)
    cv2.rectangle(frame, (w - 95, h - 55), (w - 25, h - 25), (255, 255, 255), 1)
    cv2.putText(frame, "CLEAR", (w - 82, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 🔥 લાઇવ વ્હાઇટ એરો કરન્ટ કલરના નામની નીચે
    if color_index < 4:
        ax, ay = positions[color_index]
        cv2.putText(frame, "^", (ax - 5, ay + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # --- 🖐️ હેન્ડ ટ્રેકિંગ લોજિક ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # લાઈવ હેન્ડ મેશ (Green Dots)
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                hand_dot_spec,
                hand_line_spec
            )
            
            landmarks = hand_landmarks.landmark
            x1, y1 = int(landmarks[8].x * w), int(landmarks[8].y * h)
            cx, cy = int(landmarks[9].x * w), int(landmarks[9].y * h)
            
            thumb_open = landmarks[4].x > landmarks[3].x if landmarks[17].x < landmarks[5].x else landmarks[4].x < landmarks[3].x
            index_open = landmarks[8].y < landmarks[6].y
            middle_open = landmarks[12].y < landmarks[10].y
            ring_open = landmarks[16].y < landmarks[14].y
            pinky_open = landmarks[20].y < landmarks[18].y

            fingers_count = sum([thumb_open, index_open, middle_open, ring_open, pinky_open])

            # ૧. ૫ આંગળી ઓપન -> ઇરેઝર (હથેળીની વચ્ચે)
            if fingers_count >= 4:
                color_index = 5
                cv2.circle(frame, (cx, cy), 45, (255, 255, 255), 2)
                cv2.putText(frame, "ERASER", (cx - 25, cy - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                if xp != 0 and yp != 0:
                    cv2.line(canvas, (xp, yp), (cx, cy), (0, 0, 0), 90)
                xp, yp = cx, cy

            # ૨. ૨ આંગળી ઓપન -> સિલેક્શન મોડ
            elif index_open and middle_open:
                xp, yp = 0, 0
                cv2.circle(frame, (x1, y1), 8, (255, 255, 255), cv2.FILLED)
                
                if x1 > w - 110:
                    if 20 < y1 < 100:
                        if x1 < w - 60: color_index = 0
                        else: color_index = 1
                    elif 120 < y1 < 200:
                        if x1 < w - 60: color_index = 2
                        else: color_index = 3
                    elif h - 70 < y1 < h - 15:
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # ૩. ૧ આંગળી ઓપન -> સ્મૂધ રાઇટિંગ મોડ (Thickness = 3)
            elif index_open and not middle_open:
                if color_index == 5:
                    color_index = 0
                    
                cv2.circle(frame, (x1, y1), 5, current_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(canvas, (xp, yp), (x1, y1), current_color, 3)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    # મિક્સિંગ
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Canvas Pro - Premium Video Edition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()