def soma_pa(limite):
    total = 0
    for i in range(1, limite):
        if i % 2 == 0:
            total += i

    return total


if __name__ == '__main__':
    print(soma_pa(6))