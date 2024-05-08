import os
import numpy as np
import cv2
import pyttsx3

def remove_user(user_id):
    user_images_path = os.path.join('user_images', str(user_id))
    if os.path.exists(user_images_path):
        for filename in os.listdir(user_images_path):
            file_path = os.path.join(user_images_path, filename)
            os.remove(file_path)
        os.rmdir(user_images_path)
        print(f"Carpeta del usuario {user_id} eliminada correctamente.")

    face_trained_file = os.path.join('models', 'face_trained.yml')
    if os.path.exists(face_trained_file):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(face_trained_file)

        labels = []
        faces = []
        for root, dirs, files in os.walk('user_images'):
            for dir_name in dirs:
                user_id = int(dir_name)
                user_images_path = os.path.join(root, dir_name)
                for filename in os.listdir(user_images_path):
                    labels.append(user_id)
                    img_path = os.path.join(user_images_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(img)

        new_labels = []
        new_faces = []
        for label, face in zip(labels, faces):
            if label != user_id:
                new_labels.append(label)
                new_faces.append(face)

        if len(set(new_labels)) > 1:
            recognizer.train(new_faces, np.array(new_labels))
            recognizer.save(face_trained_file)
            print(f"Usuario {user_id} eliminado del archivo face_trained.yml correctamente.")
            speak_message("Usuario eliminado correctamente.")
        else:
            print("No hay suficientes usuarios restantes para entrenar el modelo.")
            speak_message("No hay suficientes usuarios restantes para entrenar el modelo.")
    else:
        print("No se encontró el archivo face_trained.yml.")
        speak_message("No se encontró el archivo face_trained.yml.")

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

user_id_to_remove = input("Ingrese el ID del usuario a eliminar: ")

if user_id_to_remove.isdigit():
    remove_user(int(user_id_to_remove))
else:
    print("El ID del usuario debe ser un número entero.")
    speak_message("El ID del usuario debe ser un número entero.")