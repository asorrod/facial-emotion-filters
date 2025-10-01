import streamlit as st
from emotion_filter import streamlit_facial_detection
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent
APP_ICON = BASE_DIR / "resources" / "images" / "app_icon.png"


st.set_page_config(page_title = "Emotion Filter App", page_icon= APP_ICON,layout= "centered" )

def main():
    st.header("Emotion filter")
    st.text("Connect your webcam and try different expressions")

    if "run" not in st.session_state:
        st.session_state["run"] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam"):
            st.session_state["run"] = True
    with col2:
        if st.button("Stop Webcam"):
            st.session_state["run"] = False

    placeholder = st.empty()

    while st.session_state["run"]:
        pil_img = streamlit_facial_detection()
        if pil_img is not None:
            placeholder.image(pil_img, channels="RGB")
        else:
            st.warning("Webcam not found")
            break
        time.sleep(0.03)  # ~30 fps aprox.

if __name__== "__main__":
    main()