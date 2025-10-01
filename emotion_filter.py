from fer import FER
import dlib
import cv2 
import numpy as np
import imageio as io
from pathlib import Path
from PIL import Image
import os

# --------------------------------------------------------
# RESOURCES
BASE_DIR = Path(__file__).parent
LANDMARK_PATH = BASE_DIR / "resources" / "predictors" / "shape_predictor_68_face_landmarks.dat"
CLOUD_IMG_PATH = BASE_DIR / "resources" / "images" / "cloud.png"
FIRE_IMG_PATH = BASE_DIR / "resources" / "images" / "fire4_64.png"
RAIN_IMG_PATH = BASE_DIR / "resources" / "images" / "rain.png"
# --------------------------------------------------------
# DETECTORS
landmark_detector = dlib.shape_predictor(str(LANDMARK_PATH))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # desactiva GPU para TF

emotion_detector = FER(mtcnn=False)

# --------------------------------------------------------
# CAMERA
cap = cv2.VideoCapture(0)

# --------------------------------------------------------
# ANIMATIONS GLOBAL VARIABLES
frame_counter_a = 0
frame_counter_s = 0
flame_frames = []
rain_frames = []
confetti_particles = []

PIXEL_SIZE = 8


def load_frames(sprite_path, num_cols, num_rows):
    sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    sheet_h, sheet_w, _ = sprite.shape
    frame_w = sheet_w // num_cols
    frame_h = sheet_h // num_rows

    frames = []
    for row in range(num_rows):
        for col in range(num_cols):
            x_min = col * frame_w
            y_min = row * frame_h
            frame = sprite[y_min:y_min+frame_h, x_min:x_min+frame_w]
            frames.append(frame)
    return frames

flame_frames = load_frames(str(FIRE_IMG_PATH), num_cols=10, num_rows=6)
rain_frames = load_frames(str(RAIN_IMG_PATH), num_cols=10, num_rows=1)

def apply_overlay(img, overlay, x_min, y_min, x_max, y_max):
    h, w, _ = img.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return img  

    overlay = cv2.resize(overlay, (x_max - x_min, y_max - y_min))

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y_min:y_max, x_min:x_max, c] = (
            alpha_s * overlay[:, :, c] +
            alpha_l * img[y_min:y_max, x_min:x_max, c]
        )

    return img

def eye_scaling(roi_dict, eye):
    x_min, y_min, x_max, y_max = roi_dict[eye]

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    w = int((x_max - x_min) * 2)
    h = int((y_max - y_min) * 2.5)

    offset_y = int(0.20 * h)
    cy -= offset_y

    x_min_new = cx - w // 2
    x_max_new = cx + w // 2
    y_min_new = cy - h // 2
    y_max_new = cy + h // 2
    
    return x_min_new, y_min_new, x_max_new, y_max_new

def happy_filter(img, confetti):
    h, w, _ = img.shape
    if not confetti: 
        for _ in range(50):
            confetti.append({
                "x": np.random.randint(0, w),
                "y": np.random.randint(-100, 0),
                "color": tuple(np.random.randint(0, 255, 3).tolist()),
                "y_speed": np.random.randint(6, 10),
                "x_speed": np.random.choice([-1, 1]) * np.random.random()
            })

    for particle in confetti:
        particle["y"] += particle["y_speed"]
        particle["x"] += particle["x_speed"] + np.random.choice([-1, 0, 1]) * 0.5

        if particle["y"] > h:
            particle["y"] = np.random.randint(-50, 0)
            particle["x"] = np.random.randint(0, w)

        if particle["x"] > w: particle["x"] = 0
        if particle["x"] < 0: particle["x"] = w

        cv2.circle(img, (int(particle["x"]), int(particle["y"])), 5, particle["color"], -1)

    return confetti

def angry_filter(img, roi_dict):
    global frame_counter_a, flame_frames

    eyes = ["left_eye", "right_eye"]
    for eye in eyes:
        overlay = flame_frames[frame_counter_a % len(flame_frames)]
        frame_counter_a += 1

        x_min_new, y_min_new, x_max_new, y_max_new = eye_scaling(roi_dict, eye)

        apply_overlay(img, overlay, x_min_new, y_min_new, x_max_new, y_max_new)

def sad_filter(img, roi_dict):
    global frame_counter_s, rain_frames

    rain_overlay = rain_frames[frame_counter_s % len(rain_frames)]
    frame_counter_s +=1
    h, w, _ = img.shape
    apply_overlay(img, rain_overlay, 0, 0, w, h)

    x_min, y_min, x_max, y_max = roi_dict["face"]
    cloud_img = cv2.imread(str(CLOUD_IMG_PATH), cv2.IMREAD_UNCHANGED)
    cloud_resized =  cv2.resize(cloud_img, (x_max - x_min, int((y_max - y_min) * 0.4)))

    apply_overlay(img, cloud_resized, x_min, y_min - int((y_max - y_min) * 2), x_max, y_min)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img

def surprise_filter(img, roi_dict, scale_factor):
    x_min_m, y_min_m, x_max_m, y_max_m = roi_dict["mouth"]
    mouth = img[y_min_m:y_max_m, x_min_m:x_max_m]

    if mouth.size > 0:
        mouth_resized = cv2.resize(mouth, (0,0), fx = scale_factor, fy = scale_factor)

        h,w, _ = mouth.shape
        h_new, w_new, _ = mouth_resized.shape
        x_offset = x_min_m - (w_new - w) // 2
        y_offset = y_min_m - (h_new - h) // 2

        img[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = mouth_resized
    
    for eye in ["left_eye", "right_eye"]:

        x_min_e, y_min_e, x_max_e, y_max_e = roi_dict[eye]
        eye_img = img[y_min_e:y_max_e, x_min_e:x_max_e]

        if eye_img.size > 0:
            eye_resized = cv2.resize(eye_img, (0, 0), fx=scale_factor, fy=scale_factor)

            h, w, _ = eye_img.shape
            h_new, w_new, _ = eye_resized.shape
            x_offset = x_min_e - (w_new - w) // 2
            y_offset = y_min_e - (h_new - h) // 2

            img[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = eye_resized





def select_filter(img, roi_dict, emotion):

    if emotion == "happy":
        global confetti_particles
        confetti_particles = happy_filter(img, confetti_particles)   

    elif emotion == "angry":
        angry_filter(img, roi_dict)

    elif emotion == "sad":
        img = sad_filter(img, roi_dict)
        
    elif emotion == "surprise":
        surprise_filter(img, roi_dict, scale_factor=1.5)

    else: pass
    
    return img

def landmark_classificator(points, padding = 20):

    landmark_groups = {
        "face": range(0, 17),
        "left_eyebrow": range(17, 22),
        "right_eyebrow": range(22, 27),
        "nose": range(27, 36),
        "left_eye": range(36, 42),
        "right_eye": range(42, 48),
        "eyes": range(36, 48),
        "mouth": range(48, 68)
    }

    roi = {}

    for group_name, index in landmark_groups.items():
        if not index:
            continue
            
        group_points = np.array([points[i] for i in index])

        x_min, y_min = group_points.min(axis=0)
        x_max, y_max = group_points.max(axis=0)

        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding

        roi[group_name] = (x_min, y_min, x_max, y_max)        

    return roi

def facial_detection():
    while True:
        ret, img = cap.read()

        results = emotion_detector.detect_emotions(img)

        for result in results:
            (x, y, w, h) = result["box"]
            
            emotions = result["emotions"]
            main_emotion = max(emotions, key=emotions.get)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(img, main_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            roi = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            landmark = landmark_detector(img, roi)

            points = []
            for n in range(0,68):
                x = landmark.part(n).x
                y = landmark.part(n).y

                cv2.circle(img, (x, y), 2, (0,255,0), -1)
                points.append((x,y))
            
            roi_dict = landmark_classificator(points)
            img = select_filter(img, roi_dict, main_emotion)
            

        
        cv2.imshow("img",img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 

def streamlit_facial_detection():

    ret, img = cap.read()

    results = emotion_detector.detect_emotions(img)

    for result in results:
        (x, y, w, h) = result["box"]
        
        emotions = result["emotions"]
        main_emotion = max(emotions, key=emotions.get)

        #cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        #cv2.putText(img, main_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        roi = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmark = landmark_detector(img, roi)

        points = []
        for n in range(0,68):
            x = landmark.part(n).x
            y = landmark.part(n).y

            #cv2.circle(img, (x, y), 2, (0,255,0), -1)
            points.append((x,y))
        
        roi_dict = landmark_classificator(points)
        img = select_filter(img, roi_dict, main_emotion)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    return pil_img
    



if __name__ == "__main__":


    facial_detection()
