# Машинное обучение в продакшене <br/>
## Студент DS-21 Жуйков Владимир <br/>
https://data.mail.ru/profile/v.zhujkov/ <br/>

### ДЗ № 1 <br/>
#### Необходимо установить пакеты: <br/>
pip install -r requirements.txt <br/>

#### Источник данных для модели: <br/>
https://www.kaggle.com/ronitf/heart-disease-uci <br/>

#### Подготвка данных: <br/>
- нормализация <br/>
- OneHotEncoding <br/>

#### Используемые модели машинного обучения: <br/>
- логистическая регрессия <br/>
- случайный лес <br/>

#### Обучение логистической регрессии: <br/>
python ml_project/train_p.py configs/config_log_r.yml train <br/>

#### Обучение случайного леса: <br/>
python ml_project/train_p.py configs/config_rand_f.yml train <br/>

#### Результат записывается в следующую папку: <br/>
/models <br/>


#### Валидация логистической регрессии: <br/>
python ml_project/train_p.py configs/config_log_r.yml validate <br/>

#### Валидация случайного леса: <br/>
python ml_project/train_p.py configs/config_rand_f.yml validate <br/>

#### Результат записывается в следующую папку: <br/>
model/predicts.csv <br/>

#### Тестирование модели: <br/>
cd ml_project <br/>
pytest ../tests --cov <br/>





#### Самооценка: <br/>
-2. ок, +1 <br/>
-1. ок, 0 <br/>
0. ок, +2 <br/>
1. ok - EDA сделано, +2<br/>
2. ок, +2<br/>
3. ок, +2<br/>
4. ок, +3<br/>
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
