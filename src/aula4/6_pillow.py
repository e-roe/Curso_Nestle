from PIL import Image

img = Image.open("../../inputs/Monty_Python.jpg")	#Em Pillow a imagem é RGB
print(img.format)   			#PNG
print(img.size)     			#(largura, altura)
print(img.mode)     			#RGB, L (grayscale), etc.
resized = img.resize((100, 100))
img.rotate(45).show()
img.transpose(Image.FLIP_LEFT_RIGHT).show()
