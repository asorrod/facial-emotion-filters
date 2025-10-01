# Real-Time Emotion Filters

This project detects facial emotions in real-time and applies fun visual filters (fire, rain, confetti, cartoon-like effects) depending on the detected emotion.  
It uses **Python, OpenCV, dlib, FER (Facial Emotion Recognition), and Streamlit**.

---

## Features
- Real-time webcam emotion detection (happy, sad, angry, surprise, etc.)
- Dynamic filters overlayed on the face:
  - Fire for "angry"
  - Rain for "sad"
  - Confetti for "happy"
  - Enlarged eyes/mouth for "surprise"
- Streamlit interface with start/stop webcam buttons

---

## Tech Stack
- **Python 3**
- [OpenCV](https://opencv.org/) – image processing
- [dlib](http://dlib.net/) – facial landmarks
- [dlib-models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) - pre-trained model from dlib
- [FER](https://github.com/justinshenk/fer) – emotion detection
- [Streamlit](https://streamlit.io/) – interactive UI

---

## Installation
```bash
git clone https://github.com/your-username/emotion-filters.git
cd emotion-filters
pip install -r requirements.txt
