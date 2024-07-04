lst = []

for i in range(50000000):
    lst.append(i)

    if i % 1000000 == 0:
        print(i * 100 / 50000000, '%')