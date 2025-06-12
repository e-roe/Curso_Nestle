import os
import cv2
import pytesseract

os.makedirs('../../outputs/aula6/', exist_ok=True)  # Cria o diretório de saída se não existir

# Configurar o caminho do executável do Tesseract (ajuste conforme necessário)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carregar a imagem
image_path = '../../inputs/aula6/plate1.jpg'  # Substitua pelo caminho da sua imagem
image = cv2.imread(image_path)

# Converter a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização para melhorar o contraste
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite('../../outputs/aula6/plate1_thresh.jpg', thresh)  # Salvar imagem processada

# Realizar OCR na imagem processada
text = pytesseract.image_to_string(thresh, lang='eng')  # Ajuste o idioma conforme necessário

# Exibir o texto extraído
print("Texto extraído:")
print(text)

# Exibir a imagem original e a imagem processada
cv2.imshow("Imagem Original", image)
cv2.imshow("Imagem Binarizada", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()