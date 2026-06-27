import cv2
import numpy as np
import mediapipe as mp

# MediaPipe હેન્ડ ટ્રેકિંગ સેટઅપ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)

# પ્રોફેશનલ ડાર્ક અને રીચ કલર્સ (BGR Format)
colors = [
    (180, 0, 0),     # Deep Blue
    (0, 150, 0),     # Forest Green
    (0, 0, 180),     # Dark Red
    (0, 180, 180)    # Deep Yellow/Gold
]
color_index = 0 # 0: Blue, 1: Green, 2: Red, 3: Yellow, 4: RGB, 5: Eraser
hue = 0

canvas = None
xp, yp = 0, 0

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

    # --- મિનિમલ અને પ્રોફેશનલ સાઇડ UI (Small & Sleek) ---
    # પાતળું અને પ્રીમિયમ બેકગ્રાઉન્ડ પેનલ (w-80 થી w સુધી જ - એકદમ નાનું)
    cv2.rectangle(frame, (w - 70, 0), (w, h), (30, 30, 30), cv2.FILLED)
    
    # રાઉન્ડ આકારના નાના કલર ડોટ્સ / ઈન્ડિકેટર્સ
    cv2.circle(frame, (w - 35, 40), 15, (180, 0, 0), cv2.FILLED)      # Blue
    cv2.circle(frame, (w - 35, 100), 15, (0, 150, 0), cv2.FILLED)     # Green
    cv2.circle(frame, (w - 35, 160), 15, (0, 0, 180), cv2.FILLED)     # Red
    cv2.circle(frame, (w - 35, 220), 15, (0, 180, 180), cv2.FILLED)   # Yellow
    
    # નાના પ્રીમિયમ ટેક્સ્ટ બટન્સ
    cv2.rectangle(frame, (w - 65, 270), (w - 5, 310), (60, 60, 60), cv2.FILLED)
    cv2.putText(frame, "RGB", (w - 50, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.rectangle(frame, (w - 65, 330), (w - 5, 370), (10, 10, 10), cv2.FILLED)
    cv2.putText(frame, "CLEAR", (w - 55, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # કરન્ટ કલર સિલેક્શન લોજિક
    if color_index == 4:
        hue = (hue + 2) % 180
        rgb_color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        current_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
    elif color_index == 5:
        current_color = (0, 0, 0)
    else:
        current_color = colors[color_index]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # પોઇન્ટર્સ
            x1, y1 = int(landmarks[8].x * w), int(landmarks[8].y * h)   # Index Finger Tip
            
            # હાથની મધ્યમ જગ્યા (Hath ni vache - Wrist & Knuckles નું સેન્ટર)
            # Landmark 9 (Middle finger MCP) એ હથેળીની બરોબર વચ્ચેનો ભાગ ગણાય
            cx, cy = int(landmarks[9].x * w), int(landmarks[9].y * h)

            # આંગળીઓ ઓપન ચેકિંગ લોજિક
            thumb_open = landmarks[4].x > landmarks[3].x if landmarks[17].x < landmarks[5].x else landmarks[4].x < landmarks[3].x
            index_open = landmarks[8].y < landmarks[6].y
            middle_open = landmarks[12].y < landmarks[10].y
            ring_open = landmarks[16].y < landmarks[14].y
            pinky_open = landmarks[20].y < landmarks[18].y

            fingers_count = sum([thumb_open, index_open, middle_open, ring_open, pinky_open])

            # ૧. ૫ આંગળી ઓપન -> ઇરેઝર હથેળીની વચ્ચે (Hand Center Eraser)
            if fingers_count >= 4:
                color_index = 5
                
                # હવાનું ડસ્ટર (હથેળીની વચ્ચે સર્કલ બનશે)
                cv2.circle(frame, (cx, cy), 35, (200, 200, 200), 2)
                cv2.putText(frame, "ERASING", (cx - 30, cy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                if xp != 0 and yp != 0:
                    cv2.line(canvas, (xp, yp), (cx, cy), (0, 0, 0), 70)
                xp, yp = cx, cy

            # ૨. ૨ આંગળી ઓપન -> સ્લીક સાઇડ મેનૂ સિલેક્શન
            elif index_open and middle_open:
                xp, yp = 0, 0
                cv2.circle(frame, (x1, y1), 8, (255, 255, 255), cv2.FILLED) # નાનું વ્હાઇટ સિલેક્ટર પોઇન્ટ
                
                if x1 > w - 70:
                    if 25 < y1 < 55:
                        color_index = 0
                    elif 85 < y1 < 115:
                        color_index = 1
                    elif 145 < y1 < 175:
                        color_index = 2
                    elif 205 < y1 < 235:
                        color_index = 3
                    elif 270 < y1 < 310:
                        color_index = 4
                    elif 330 < y1 < 370:
                        canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear

            # ૩. ૧ આંગળી ઓપન -> પ્રિસાઇઝ ફાઇન રાઇટિંગ (Nano Width)
            elif index_open and not middle_open:
                cv2.circle(frame, (x1, y1), 4, current_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if color_index == 5:
                    cv2.line(canvas, (xp, yp), (cx, cy), (0, 0, 0), 70)
                else:
                    # વિડ્થ માત્ર '2' રાખી છે જેથી અક્ષરો એકદમ શાર્પ, પ્રીમિયમ અને પ્રોફેશનલ લાગે
                    cv2.line(canvas, (xp, yp), (x1, y1), current_color, 2)

                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    # કેનવાસ અને કેમેરા ફ્રેમ મિક્સિંગ
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    # આઉટપુટ
    cv2.imshow("Air Canvas Pro - Premium UI", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()