import cv2
import numpy as np
from skimage.metrics import structural_similarity # instalar com: pip install scikit-image

print(cv2.__version__)
# Lê as duas imagens a serem comparadas
first = cv2.imread(r'C:\Roe\Stepps\projs\Curso_Nestle\src\mentoria2\scn1.png')
second = cv2.imread(r'C:\Roe\Stepps\projs\Curso_Nestle\src\mentoria2\scn2.png')

img_upper = cv2.hconcat([first, second])
cv2.imshow('Diferencas', cv2.resize(img_upper, (0, 0), fx=0.8, fy=0.8))
cv2.waitKey(0)

# Converte as imagens para escala de cinza
first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

# Calcula o índice de similaridade estrutural (SSIM) entre as duas imagens
score, diff = structural_similarity(first_gray, second_gray, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))

# A imagem diff mostra as diferenças entre as imagens, em float, então convertemos para uint8
diff = (diff * 255).astype("uint8")

# Aplica um limiar (threshold) para destacar as regiões diferentes
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Cria uma máscara e uma cópia da segunda imagem para destacar as diferenças
mask = np.zeros(first.shape, dtype='uint8')
filled = second.copy()

# Para cada contorno encontrado, desenha um retângulo nas regiões diferentes
for c in contours:
    area = cv2.contourArea(c)
    if area > 100:  # Ignora pequenas diferenças/ruídos
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(first, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(second, (x, y), (x + w, y + h), (36, 12, 255), 2)

# Junta as duas imagens lado a lado para visualização
img_upper = cv2.hconcat([first, second])
cv2.imshow('Diferencas', cv2.resize(img_upper, (0, 0), fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.imwrite('resultado.png', img_upper)