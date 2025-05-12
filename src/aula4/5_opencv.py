import cv2

def read_image():
    img = cv2.imread('../../inputs/garrafas.png')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Grey', grey)
    cv2.imshow('HSV', hsv)
    cv2.imshow('RGB', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def split():
    img = cv2.imread('../../inputs/Monty_Python.jpg')
    b, g, r = cv2.split(img)  # ou b = img[:, :, 0]
    cv2.imshow('Red', r)
    cv2.imshow('Green', g)
    cv2.imshow('Blue', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    g = g * 0
    b = b * 0
    img = cv2.merge([b, g, r])
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.imwrite('../../outputs/red.jpg', img)



if __name__ == '__main__':
    split()
