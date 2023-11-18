import cv2
import numpy as np
import os


# Функция проверки изображений
def image_definition_by_cyclone(image):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение фильтрации для выделения текстур
    filtered_image = cv2.Canny(gray_image, 100, 200)

    # Поиск контуров
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проверка каждой текстуры на схожесть с окружностью
    cyclone_found = False
    for contour in contours:
        # Приближение контура к форме окружности
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)

        # Если контур имеет форму окружности, проверяем цвет текстуры
        if len(approx) > 7:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y + h, x:x + w]
            average_color = np.mean(roi, axis=(0, 1))
            if np.all(average_color > 200):
                cyclone_found = True
                break

    # Если контур похож на циклон переходим ко второй проверке
    if cyclone_found:
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Преобразование изображения из BGR в HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Создание маски для синего цвета
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Поиск синих объектов
        blue_objects = cv2.bitwise_and(image, image, mask=blue_mask)

        # Проверка наличия синих объектов
        if np.any(blue_objects):
            print("Проверка пройдена!\n")

            return 1
        else:
            print("Проверка не пройдена!\n")
            return 0
    else:
        print("Проверка не пройдена!\n")
        return 0



# Функция обработки изображений
def findPossibleCyclone(image):

    # Инициализация координат центра циклона
    global cX, cY

    # Получение высоты и ширины исходного изображения
    height, width, _ = image.shape

    # Изменение размера изображения
    resized_image = cv2.resize(image, (800, 800))

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Применение фильтра Гаусса для сглаживания изображения
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение алгоритма Кэнни для выделения границ
    edges = cv2.Canny(blurred, 400, 100)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем черное изображение того же размера, что и исходное
    contour_image = np.zeros(gray.shape, dtype=np.uint8)

    # Рисуем контуры на черном изображении
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    # Фильтрация контуров, чтобы оставить только объемные и круглые облака
    cyclone_contours = []
    max_radius = 0
    max_solidity = 0
    most_circlelike_cyclone = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.2 * perimeter, True)

        # Определение минимально охватывающего круга контура
        (_, radius) = cv2.minEnclosingCircle(contour)

        # Вычислим solidity контура
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area

        # Вычислим соотношение сторон вписанного прямоугольника
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Проверим, является ли контур круглым и не вытянутым
        if len(approx) > 0 and radius > 10 and 0.5 < aspect_ratio < 1.5 and solidity > 0.23:
            if radius > max_radius:
                max_radius = radius
                most_circlelike_cyclone = contour

    if most_circlelike_cyclone is not None:
        print("Вероятно, циклон найден!")
        cyclone_contours.append(most_circlelike_cyclone)
    else:
        print("Циклон не найден.")

    # Обводим квадратом самый круглый и большой контур, который предположительно является циклоном
    for contour in cyclone_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(resized_image, (x - 50, y - 80), (x + w + 150, y + h + 50), (0, 0, 255), 2)
        mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x - 50, y - 80), (x + w + 150, y + h + 50), (255), -1)

        # Удаление объектов за пределами прямоугольника
        cleared_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

        contours_rect, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация контуров для получения прямоугольных форм
        for contour in contours_rect:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4:
                # Создание маски для прямоугольника
                mask_rect = np.zeros(cleared_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_rect, [contour], 0, (255), -1)

                gray = cv2.cvtColor(cleared_image, cv2.COLOR_BGR2GRAY)

                # Применение порогового фильтра для выделения светлых областей
                _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

                # Нахождение контуров на изображении
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Нахождение самого большого контура, схожего с окружностью
                max_contour = None
                max_contour_area = 0
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                    if len(approx) > 4:
                        area = cv2.contourArea(contour)
                        if area > max_contour_area:
                            max_contour_area = area
                            max_contour = contour
                            # Создание изображения с выделенным контуром
                        if max_contour is not None:
                            x, y, w, h = cv2.boundingRect(max_contour)
                            cv2.drawContours(resized_image, [max_contour], 0, (255, 0, 0), 2)

    x1 = max(x - 50, 0)
    y1 = max(y - 80, 0)
    x2 = min(x + w + 150, resized_image.shape[1])
    y2 = min(y + h + 50, resized_image.shape[0])

    # Обрезание изображения
    cropped_image = resized_image[y1:y2, x1:x2]

    # Вычисляем центр масс контура циклона
    # Заменяем этот код на новый, более точный способ вычисления центра


    # вычисление моментов для каждого контура
    M = cv2.moments(most_circlelike_cyclone)

    # вычисляем центр контура
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        cX, cY = 0, 0

    center = (cX, cY)

    cv2.circle(resized_image, center, 5, (0, 0, 255), -1)

    print("Координаты центра циклона:", center)
    return '-', cX, cY, width, height
    #return resized_image


# Функция получения путей к файлам в определенной директориве
def find_files_by_extension(folder_path, extension):
    file_list = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(extension):
                file_list.append(os.path.join(foldername, filename))
    return file_list


# Директория в которой проверяются файлы
folder_path = "images/"
# Расширение файлов для проверки
extension = ".jpg"

files_with_extension = find_files_by_extension(folder_path, extension)

# Счетчик количества изображений на которых возможно имеется циклон

cycloneCounter = 0

# Массив отсортированных изображений
sortedImages = []


for file_path in files_with_extension:
    image = cv2.imread(file_path)
    if image_definition_by_cyclone(image) == 1:
        sortedImages.append(file_path)
        cycloneCounter += 1


for path in sortedImages:
    image = cv2.imread(path)
    my_file = open("ResInfo" + str(cycloneCounter) + ".txt", "w+")
    my_file.write(str(findPossibleCyclone(image)))
    my_file.close()
    cycloneCounter += 1


cv2.waitKey(0)
cv2.destroyAllWindows()