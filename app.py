import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def position_data(lmlist):
    global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip  = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])
def draw_line(p1, p2, size=5):
    cv2.line(frame, p1, p2, (50,50,255), size)
    cv2.line(frame, p1, p2, (255, 255, 255), round(size / 2))

def calculate_distance(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght


while True:
    _, frame = video.read()
    frame=cv2.flip(frame,1)
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lmlist = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                x, y = int(lm.x*w), int(lm.y*h)
                lmlist.append([x,y])
                # cv2.circle(frame, (x,y), 6, (50,50,255), 3)
            position_data(lmlist)
            palm = calculate_distance(wrist, index_mcp)
            distance = calculate_distance(index_mcp, pinky_tip)
            ratio = distance/palm
            print(ratio)
            if (0.5<ratio<0.8):
                draw_line(wrist, thumb_tip)
                draw_line(wrist, index_tip)
                draw_line(wrist, pinky_tip)
                draw_line(wrist, midle_tip)
                draw_line(wrist, ring_tip)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()