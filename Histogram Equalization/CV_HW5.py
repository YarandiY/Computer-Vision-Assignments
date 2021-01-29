import numpy as np
from matplotlib import pyplot
import cv2 as cv
import random


def enhance_image(data):
    minimum = np.min(data)
    maximum = np.max(data)
    result = (data - minimum) / (maximum - minimum)
    result = result * 255
    return result.astype(int)


def find_cells(data, color, pixels):
    for row in range(0, np.shape(data)[0]):
        for col in range(0, np.shape(data)[1]):
            if data[row][col] == color:
                pixels.append([row, col])
    return pixels


data = enhance_image(cv.imread('images.jpg', 0))
number_of_pixels = np.shape(data)[0] * np.shape(data)[1]
ps = int(number_of_pixels / 256) + 1
frequency = np.zeros(256)
for i in range(0, len(data)):
    for j in range(0, len(data)):
        frequency[data[i][j]] += 1
print(ps)
for i in range(0, len(frequency) - 1):
    print(i)
    p = []
    if frequency[i] > ps:
        p = find_cells(data, i, p)
        for j in range(0, int(frequency[i] - ps)):
            random_index = random.choice(p)
            p.remove(random_index)
            frequency[data[random_index[0]][random_index[1]]] -= 1
            new_color = random.randint(i + 1, 255)
            frequency[new_color] += 1
            data[random_index[0]][random_index[1]] = new_color
    else:
        if frequency[i] == ps:
            continue
        p = find_cells(data, i + 1, p)
        temp = i + 2
        while len(p) < int(ps - frequency[i]):
            p = find_cells(data, temp, p)
            temp += 1
        for j in range(0, int(ps - frequency[i])):
            random_index = random.choice(p)
            p.remove(random_index)
            frequency[data[random_index[0]][random_index[1]]] -= 1
            frequency[i] += 1
            data[random_index[0]][random_index[1]] = i
    print(data)
    if i % 30 == 0:
        pyplot.plot(frequency)
        pyplot.show()
pyplot.plot(frequency)
pyplot.show()
cv.imwrite('result2.jpg', data)
