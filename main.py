import cv2
import os
import numpy as np
import pyttsx3

url = 'http://192.168.1.7:4747/video'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

def capture_and_train(user_id):
    user_images_path = os.path.join('user_images', str(user_id))
    os.makedirs(user_images_path, exist_ok=True)

    cap = cv2.VideoCapture(url)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(user_images_path, f'user_{count}.jpg'), roi_gray)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

        cv2.imshow('Capture Images', frame)
        if cv2.waitKey(20) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

    train_model()
    speak_message("Modelo entrenado con éxito. Ahora puedes iniciar sesión.")

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def validate_user_id(user_id):
    try:
        int(user_id)
        return True
    except ValueError:
        return False

def train_model():
    faces = []
    labels = []
    
    for root, dirs, files in os.walk('user_images'):
        for dir_name in dirs:
            user_id = int(dir_name)
            user_images_path = os.path.join(root, dir_name)
            for filename in os.listdir(user_images_path):
                img_path = os.path.join(user_images_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(user_id)

    recognizer.train(faces, np.array(labels))
    recognizer.save(os.path.join('models', 'face_trained.yml'))
    print('Modelo entrenado con éxito.')

user_id = input("Ingrese el ID del nuevo usuario: ")
if validate_user_id(user_id):
    capture_and_train(user_id)
else:
    print("El ID del usuario debe ser un número entero.")
    speak_message("El ID del usuario debe ser un número entero.")