# DetCyclone
Файл TropicalCylconesDef.py представляет собой 3 функции, занимающиеся проверкой наличия циклона на изображении,
его нахождение по некоторым признакам описанным в коментариях к коду, а так же загрузка изображений определенного формата (jpg). 
Файл test.py представялет собой заготовку к  еще одной проверке, ее суть заключается в получении всех цветов массива элементами которого являются 3 цвета каждого пикслея,
данный алгортм обрабатывает эти значения перобразовывая в hex, данная проверка определяет является ли иозбражение фотографий, например если у нас будет изображение на котором
находится белый круг и синий фон за ним, что бы софт не посчитал данный объект циклоном необходимо искулючить вариант когда изображение имеет цвет всех пикселей внутри объекта одинаковым. Не советую использовать изображения лежащие в папке images тк эта папка создана искулючительно для тестов, для проверки работы необходимо использовать предоставленные 
Сириусом снимики из космоса.

