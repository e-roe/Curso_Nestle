import cv2

# Carregar o classificador pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar a imagem
imagem = cv2.imread("../inputs/faces.jpg")

# Converter para escala de cinza (requisito para Haar Cascade)
grey = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar faces na imagem
faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenhar retângulos verdes ao redor das faces detectadas
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir a imagem com as detecções
cv2.imshow("Detecção de Faces", imagem)
cv2.imwrite("../outputs/detected_faces.jpg", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
