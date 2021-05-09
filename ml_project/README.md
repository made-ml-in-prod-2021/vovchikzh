Машинное обучение в продакшене
Студент DS-21 Жуйков Владимир
https://data.mail.ru/profile/v.zhujkov/

ДЗ № 1
Необходимо установить пакеты:
pip install -r requirements.txt

Источник данных для модели: 
https://www.kaggle.com/ronitf/heart-disease-uci

Подготвка данных:
- нормализация
- OneHotEncoding

Используемые модели машинного обучения: 
- логистическая регрессия
- случайный лес

Обучение логистической регрессии:
python ml_project/train.py configs/config_log_r.yml train

Обучение случайного леса:
python ml_project/train.py configs/config_rand_f.yml train

Результат записывается в следующую папку:
/models


Валидация логистической регрессии:
python ml_project/train.py configs/config_log_r.yml validate

Валидация случайного леса: 
python ml_project/train.py configs/config_rand_f.yml validate

Результат записывается в следующую папку:
model/predicts.csv


Тестирование модели: 
cd ml_project
pytest ../tests --cov





Самооценка: 

-2. ок, +1
-1. ок, 0
0. ок, +2
1. ok - EDA сделано, +2
2. ок, +2
3. ок, +2
4. ок, +3
5. ок - с помощью hypothesis, +3
6. ок - две корректные конфигурации для логистической регрессии и для случайного леса, +3
7. ок, +3
8. ок, +3
9. ок, +3
10. ок, +3
11. нет, 0
12. нет, 0
13.ок, +1
Итого: 31
