# pip install cmake
# conda install -c conda-forge dlib
# pip install face_recognition
import face_recognition
import cv2

largura = 1920 // 2
altura = 1080 // 2
# Configuração do codec para salvar o vídeo
fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')

# Criar o objeto VideoWriter para salvar o vídeo
out = cv2.VideoWriter(r'../../outputs/reconhecimento_facial.mp4', fourcc, 10, (largura, altura))

# Carregar a imagem de referência e codificar o rosto
reference_image_path = '../../inputs/aula6/reference_face.png'
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Abrir o vídeo para reconhecimento facial
video_path = '../../inputs/aula6/reconhecimento_facial.mp4'  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (largura, altura))  # Redimensionar o frame para largura e altura especificadas
    # Converter o frame para RGB (face_recognition usa RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Localizar e codificar rostos no frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar o rosto detectado com o rosto de referência
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            label = "Match"
            color = (0, 255, 0)  # Verde para correspondência
        else:
            label = "No Match"
            color = (0, 0, 255)  # Vermelho para não correspondência

        # Desenhar o retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Exibir o frame
    cv2.imshow("Reconhecimento Facial", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()