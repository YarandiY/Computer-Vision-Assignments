import numpy as np
import cv2 as cv
import math


def salt_pepper(image):
    new_image = image
    number = int(np.shape(image)[0]*np.shape(image)[1] * 0.1)
    for i in range(0, number):
        x = np.random.randint(0, np.shape(image)[0])
        y = np.random.randint(0, np.shape(image)[1])
        if image[x][y] < (256/2):
            new_image[x][y] = 0
        else:
            new_image[x][y] = 255
    return new_image


def median_filter(data):
    result = data
    for i in range(0, np.shape(data)[0]):
        for j in range(0, np.shape(data)[1]):
            if data[i][j] == 0 or data[i][j] == 255 :
                neighbors = np.sort(get_neighbors(data, i, j))
                median = int(len(neighbors)/2)
                result[i][j] = neighbors[median]
    return result

ni_x = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
ni_y = [1, 1, 1, 0, 0, 0, -1, -1, -1]


def get_neighbors(data, x, y):
    result = []
    counter = 0
    size_x = np.shape(data)[0]
    size_y = np.shape(data)[1]
    for i in range(len(ni_x)):
        new_x = x + ni_x[i]
        new_y = y + ni_y[i]
        if -1 < new_x < size_x and -1 < new_y < size_y:
            result.append(data[new_x][new_y])
            counter += 1
    return np.array(result)


def varians(data):
    m = mean(data)
    result = 0
    mn = np.shape(data)[0]*np.shape(data)[1]
    for i in range(0, np.shape(data)[0]):
        for j in range(0, np.shape(data)[1]):
            result += ((data[i][j] - m)**2)/mn
    return result


def mean_2D(vector):
    result = 0
    for i in range(len(vector)):
        result += vector[i]
    return result/len(vector)


def mean(data):
    result = 0;
    mn = np.shape(data)[0]*np.shape(data)[1]
    for i in range(0, np.shape(data)[0]):
        for j in range(0, np.shape(data)[1]):
            result += data[i][j]
    return result/mn


def gaussian(data):
    # enheraf = math.sqrt(varians(data))
    # mea = mean(data)
    # makhraj = (2 * math.pow(enheraf, 2))
    # makhraj2 = (enheraf * math.sqrt(2*math.pi))
    # tavan = -1 * np.power(data - mea, 2) / (makhraj)
    # print(tavan)
    # new_data = np.exp(tavan) / (makhraj2)
    # print(new_data)
    # new_data = (new_data + 0.5) * 256;
    # print("^^^^", new_data)
    # return new_data
    data_copy = data
    gaussian_d = np.random.normal(mean(data), 20, (data_copy.shape[0], data_copy.shape[1]))
    gaussian_d = gaussian_d.reshape(data_copy.shape[0], data_copy.shape[1]).astype('uint8')
    print(gaussian_d)
    result = cv.add(data_copy, gaussian_d)
    return result - 100


def smoothing_filter(data):
    result = data
    for i in range(0, np.shape(data)[0]):
        for j in range(0, np.shape(data)[1]):
            neighbors = get_neighbors(data, i, j)
            m = mean_2D(neighbors)
            result[i][j] = m
    return result


#salt&pepper noise
data_image = cv.imread('images.jpg', 0)
sp = salt_pepper(data_image)
cv.imwrite('salt&pepper_noise.png', sp)

#smoothing filter
sf = smoothing_filter(sp)
cv.imwrite('salt&pepper_smoothing_filter.png', sf)

#median filter
sp = cv.imread('salt&pepper_noise.png', 0)
mf = median_filter(sp)
cv.imwrite('salt&pepper_median_filter.png', mf)

#gaussian noise
data_image = cv.imread('images.jpg', 0)
gn = gaussian(data_image)
cv.imwrite('gaussian.png', gn)

#smoothing filter
gn = cv.imread('gaussian.png', 0)
sf = smoothing_filter(gn)
cv.imwrite('gaussian_smoothing_filter.png', sf)

#median filter
gn = cv.imread('gaussian.png', 0)
mf = median_filter(gn)
cv.imwrite('gaussian_median_filter.png', mf)