import cv2
import numpy as np

KERNEL_SIZE = 5

if __name__ == '__main__':
    img = cv2.imread('../../inputs/frame.png')
    cv2.imshow('', img)
    cv2.waitKey(0)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', grey)
    cv2.waitKey(0)

    flipped = cv2.flip(img, 1)
    cv2.imshow('', flipped)
    cv2.waitKey(0)

    flipped = cv2.flip(img, 0)
    cv2.imshow('', flipped)
    cv2.waitKey(0)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    cv2.imshow('', sharpened)
    cv2.waitKey(0)

    # Gerar ruído Gaussiano
    mean = 0
    stddev = 25  # Desvio padrão do ruído
    gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noisy_image = cv2.add(img.astype(np.float32), gaussian_noise)

    # Manter os valores dentro do intervalo válido [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cv2.imshow('', noisy_image)
    cv2.waitKey(0)

    kernel_h = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
    kernel_h[int((KERNEL_SIZE - 1) / 2), :] = np.ones(KERNEL_SIZE)
    kernel_h /= KERNEL_SIZE
    horizonal_blurred = cv2.filter2D(img, -1, kernel_h)
    cv2.imshow('', horizonal_blurred)
    cv2.waitKey(0)

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsvImg)
    # Increase the saturation (e.g., by 10%)
    s = np.clip(s * .5, 0, 255).astype(np.uint8)
    # Merge the channels back
    enhanced_hsv = cv2.merge([h, s, v])
    sat_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('', sat_image)
    cv2.waitKey(0)

    alpha = 2.  # Contrast control
    beta = 4  # Brightness control
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow('', adjusted)
    cv2.waitKey(0)

    # Obter as dimensões da imagem
    (h, w) = img.shape[:2]

    # Definir o ponto central para a rotação
    center = (w // 2, h // 2)

    # Criar a matriz de rotação
    angle = 35  # Ângulo de rotação
    scale = 1.0  # Escala da imagem
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Aplicar a rotação
    rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))

    # Exibir a imagem rotacionada
    cv2.imshow('', rotated_image)
    cv2.waitKey(0)

    crop = img[150:650, 70:800]
    cv2.imshow('', crop)
    cv2.waitKey(0)

    cv2.destroyAllWindows()