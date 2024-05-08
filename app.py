import cv2
import os
import numpy as np
from threading import Thread
from time import sleep
import pyttsx3
from customtkinter import CTk, CTkLabel, CENTER

url = 'http://192.168.1.7:4747/video'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    speak_message("Accediendo al modelo, por favor espere...")
    faces = []
    labels = []
    
    for user_folder in os.listdir('user_images'):
        user_id = int(user_folder)
        for filename in os.listdir(os.path.join('user_images', user_folder)):
            img_path = os.path.join('user_images', user_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(user_id)

    recognizer.train(faces, np.array(labels))
    recognizer.save(os.path.join('models', 'face_trained.yml'))
    print('Modelo entrenado con éxito.')
    speak_message("Modelo entrenado con éxito.")

def recognize_user(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 50:
            return f"Usuario {label}", (x, y, x+w, y+h)

    return "Desconocido", None

def show_user_detected_message(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(frame, 'Usuario detectado', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

def login():
    recognizer.read(os.path.join('models', 'face_trained.yml'))

    cap = cv2.VideoCapture(url)
    user_detected = False

    cv2.namedWindow('Reconocimiento de Usuarios', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Reconocimiento de Usuarios', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = cv2.getTickCount()
    scan_effect = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scan_effect:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if cv2.getTickCount() - start_time > 3 * cv2.getTickFrequency():
            scan_effect = False

        if not scan_effect:
            username, bbox = recognize_user(frame)
            if username != "Desconocido":
                user_detected = True
                color = (0, 255, 0)
                show_user_detected_message(frame)
            else:
                user_detected = False
                color = (0, 0, 255)
                cv2.putText(frame, 'Desconocido', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if bbox is not None:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            cv2.imshow('Reconocimiento de Usuarios', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if user_detected:
            break

    cap.release()
    cv2.destroyAllWindows()

    if user_detected:
        speak_message("¡Bienvenido!")
        welcome_window = CTk()
        welcome_window.title("¡Bienvenido!")
        welcome_window.attributes("-topmost", True)
        window_width = 300
        window_height = 100
        welcome_window.geometry(f"{window_width}x{window_height}")
        
        label = CTkLabel(welcome_window, text="¡Bienvenido!", fg_color="transparent", font=("Segoe UI", 20))
        label.pack(expand=True, fill="both")
        welcome_window.mainloop()

def start_login_with_loading_screen():
    loading_window = CTk()
    loading_window.title("Cargando modelo ...")
    loading_window.attributes("-topmost", True)
    
    loading_window._set_appearance_mode("light")
    window_width = 300
    window_height = 100
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    x_coordinate = (screen_width - window_width) / 2
    y_coordinate = (screen_height - window_height) / 2
    loading_window.geometry(f"{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}")
    
    loading_label = CTkLabel(loading_window, text="Accediendo al modelo, por favor espere...", fg_color="transparent", anchor=CENTER)
    loading_label.pack(expand=True, fill="both")
    
    def start_login():
        train_model()
        sleep(5)
        loading_window.destroy()
        login_thread = Thread(target=login)
        login_thread.start()

    loading_window.after(100, start_login)
    loading_window.mainloop()

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

start_login_with_loading_screen()