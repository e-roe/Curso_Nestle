texto = "  Olá, Mundo!  "

print(texto.lower())

print(texto.upper())

print(texto.strip())

print(texto.replace("a", "o"))

palavras = texto.strip().split()
print(palavras)

print(texto.startswith("  Olá"))

print(texto.endswith("!  "))
