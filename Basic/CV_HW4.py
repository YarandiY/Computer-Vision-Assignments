import numpy as np
from matplotlib import pyplot
import cv2 as cv


def enhance_image(data):
    minimum = np.min(data)
    maximum = np.max(data)
    result = (data - minimum) / (maximum - minimum)
    result = result * 255
    return result.astype(int)


def draw_histogram(data):
    frequency = np.zeros(256)
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            frequency[data[i][j]] += 1
    pyplot.plot(frequency)
    pyplot.show()
    return frequency


initial_image_data = cv.imread('images.jpg', 0)
result_image_data = enhance_image(initial_image_data)
draw_histogram(initial_image_data)
draw_histogram(result_image_data)
cv.imwrite('result.temp.jpg', result_image_data)
cv.destroyAllWindows()
