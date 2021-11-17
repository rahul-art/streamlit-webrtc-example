from webcam_component import webcam

captured_image = webcam()
if captured_image is not None:
   st.image(captured_image) 
