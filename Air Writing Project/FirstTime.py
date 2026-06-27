import cv2
import numpy as np
import mediapipe as mp

# MediaPipe હેન્ડ ટ્રેકિંગ સેટઅપ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# કલર ઓપ્શન્સ અને કન્વાસ સેટઅપ
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow
color_index = 0

# ડ્રોઇંગ માટે બ્લેન્ક વ્હાઇટ સ્ક્રીન (કેનવાસ)
canvas = None

# અગાઉના પોઇન્ટના કો-ઓર્ડિનેટ્સ સ્ટોર કરવા
xp, yp = 0, 0

# વેબકેમ શરૂ કરો
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # ફ્રેમને ફ્લિપ કરો (જેથી અરીસા જેવું કામ કરે)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # કેનવાસ ઇનિશિયલાઇઝ કરો જો ન થયો હોય તો
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR થી RGB માં કન્વર્ટ કરો
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # ઉપરના મેનૂ બાર (બટનો) સ્ક્રીન પર ડ્રો કરો
    cv2.rectangle(frame, (10, 10), (110, 70), (255, 0, 0), cv2.FILLED) # Blue
    cv2.rectangle(frame, (120, 10), (220, 70), (0, 255, 0), cv2.FILLED) # Green
    cv2.rectangle(frame, (230, 10), (330, 70), (0, 0, 255), cv2.FILLED) # Red
    cv2.rectangle(frame, (340, 10), (440, 70), (0, 255, 255), cv2.FILLED) # Yellow
    cv2.rectangle(frame, (450, 10), (560, 70), (200, 200, 200), cv2.FILLED) # Eraser
    cv2.rectangle(frame, (570, 10), (670, 70), (0, 0, 0), cv2.FILLED) # Clear All

    cv2.putText(frame, "CLEAR", (585, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "ERASER", (470, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # આંગળીઓના પોઇન્ટ્સ મેળવો
            landmarks = hand_landmarks.landmark
            
            # Index finger (તર્જની) અને Middle finger (મધ્યમા) ના ટેરવાં
            x1, y1 = int(landmarks[8].x * w), int(landmarks[8].y * h)
            x2, y2 = int(landmarks[12].x * w), int(landmarks[12].y * h)

            # આંગળીઓ ખુલ્લી છે કે બંધ તે ચેક કરો
            index_open = landmarks[8].y < landmarks[6].y
            middle_open = landmarks[12].y < landmarks[10].y

            # ૧. સિલેક્શન મોડ: જો બંને આંગળીઓ (Index & Middle) ઉપર હોય
            if index_open and middle_open:
                xp, yp = 0, 0 # ડ્રોઇંગ અટકાવો
                cv2.circle(frame, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                
                # બટન ક્લિક્સ ચેક કરો (જ્યારે આંગળીઓ મેનૂ બારમાં હોય)
                if y1 < 70:
                    if 10 < x1 < 110:
                        color_index = 0 # Blue
                    elif 120 < x1 < 220:
                        color_index = 1 # Green
                    elif 230 < x1 < 330:
                        color_index = 2 # Red
                    elif 340 < x1 < 440:
                        color_index = 3 # Yellow
                    elif 450 < x1 < 560:
                        color_index = 4 # Eraser Mode
                    elif 570 < x1 < 670:
                        canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear screen

            # ૨. ડ્રોઇંગ મોડ: માત્ર Index Finger ઉપર હોય (Middle Finger બંધ હોય)
            elif index_open and not middle_open:
                cv2.circle(frame, (x1, y1), 10, colors[color_index if color_index < 4 else 0], cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if color_index == 4: # જો ઇરેઝર સિલેક્ટ હોય
                    cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), 50) # જાડી લાઈનથી ઇરેઝ થશે
                    cv2.circle(frame, (x1, y1), 25, (255, 255, 255), 2)
                else: # જો કોઈ કલર સિલેક્ટ હોય તો ડ્રોઇંગ થશે
                    cv2.line(canvas, (xp, yp), (x1, y1), colors[color_index], 5)

                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    # કેનવાસને ઓરિજિનલ વિડિયો ફ્રેમ સાથે મિક્સ કરો
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    # આઉટપુટ ડિસ્પ્લે
    cv2.imshow("Air Drawing System", frame)
    
    # 'q' દબાવવાથી પ્રોગ્રામ બંધ થશે
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()