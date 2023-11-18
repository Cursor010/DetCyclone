import cv2
import numpy as np

count_dict = {}

def get_hex(n):
    ans = str(hex(n))
    if len(ans) == 1:
        ans = '0' + ans
    return ans

# Загрузка изображения
image = cv2.imread('images/cyclone1.jpg')

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение порогового фильтра для выделения ярких объектов
_, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Нахождение контуров на изображении
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Нахождение самого большого контура, схожего с окружностью
max_contour = None
max_contour_area = 0
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) > 4:  # Проверка, что контур схож с окружностью
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            max_contour = contour
            if max_contour is not None:
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.drawContours(image, [max_contour], 0, (255, 0, 0), 2)



# Создание массива с цветами каждого пикселя в круге
if max_contour is not None:
    # Создаем маску для самого большого контура
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], 0, (255), -1)

    # Извлекаем цвета пикселей в круге
    circle_pixels = image[np.where(mask != 0)]
    result_array = circle_pixels [~np.all(circle_pixels  == [255, 0, 0], axis=1)]

    # Выводим массив цветов
    print("Массив цветов каждого пикселя в круге:\n", result_array, len(result_array))


def check(pixels):
    count_dict = {}

    def get_hex(n):
        ans = str(hex(n))[2:]
        if len(ans) == 1:
            ans = '0' + ans
        return ans

    for pixel in pixels:
        color = get_hex(pixel[0]) + get_hex(pixel[1]) + get_hex(pixel[2])
        try:
            count_dict[color] += 1
        except KeyError:
            count_dict[color] = 1

    return count_dict

print(check(circle_pixels))
# Отображение результата
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()