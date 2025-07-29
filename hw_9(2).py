import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Заголовок приложения
st.title("Прогнозирование стоимости недвижимости")
st.write("Введите параметры недвижимости для получения прогноза цены")

# Функция для создания и обучения модели (если файл модели не существует)
def create_and_train_model():
    # Создаем демонстрационные данные, если файл Housing.csv не найден
    data_str = """price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
13300000,7420,4,2,3,yes,no,no,no,yes,2,yes,furnished
12250000,8960,4,4,4,yes,no,no,no,yes,3,no,furnished
12250000,9960,3,2,2,yes,no,yes,no,no,2,yes,semi-furnished
12215000,7500,4,2,2,yes,no,yes,no,yes,3,yes,furnished
11410000,7420,4,1,2,yes,yes,yes,no,yes,2,no,furnished
10850000,7500,3,3,1,yes,no,yes,no,yes,2,yes,semi-furnished
10150000,8580,4,3,4,yes,no,no,no,yes,2,yes,semi-furnished
10150000,16200,5,3,2,yes,no,no,no,no,0,no,unfurnished
6125000,6420,3,1,3,yes,no,yes,no,no,0,yes,unfurnished
6107500,3240,4,1,3,yes,no,no,no,no,1,no,semi-furnished
6090000,5400,3,1,2,yes,no,no,no,no,2,no,unfurnished
6055000,3600,3,1,1,yes,no,no,no,no,0,no,unfurnished
6020000,4600,4,1,2,yes,no,no,no,yes,2,no,semi-furnished
5985000,6600,3,1,1,yes,no,no,no,no,0,no,unfurnished
5950000,6600,3,1,1,yes,no,no,no,no,0,no,unfurnished
5950000,6750,3,1,1,yes,no,no,no,no,0,no,unfurnished
5950000,4600,4,1,2,yes,no,no,no,yes,2,no,semi-furnished
5915000,6615,3,1,1,yes,no,no,no,no,0,no,unfurnished
5880000,6000,3,1,1,yes,no,no,no,no,0,no,unfurnished
5880000,6000,3,1,1,yes,no,no,no,no,0,no,unfurnished"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(data_str))
    
    # Предобработка данных
    df_processed = df.copy()
    
    # Преобразуем категориальные признаки yes/no в 1/0
    yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in yes_no_columns:
        df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
    
    # Преобразуем категориальный признак furnishingstatus с помощью LabelEncoder
    label_encoder = LabelEncoder()
    df_processed['furnishingstatus'] = label_encoder.fit_transform(df_processed['furnishingstatus'])
    
    # Определение признаков и целевой переменной
    features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
               'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    target = 'price'
    
    X = df_processed[features]
    y = df_processed[target]
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X, y)
    
    # Сохранение модели и энкодера
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, label_encoder

# Проверяем, существует ли файл модели, если нет - создаем его
if not os.path.exists('model.pkl'):
    with st.spinner('Создание и обучение модели...'):
        model, label_encoder = create_and_train_model()
        st.success('Модель успешно создана и обучена!')
else:
    # Загружаем модель и энкодер
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Создаем и обучаем энкодер заново для демонстрации
    _, label_encoder = create_and_train_model()

# Создаем форму для ввода параметров недвижимости
st.header("Параметры недвижимости")

# Числовые параметры
area = st.number_input("Площадь (кв. футов)", min_value=100, max_value=50000, value=5000)
bedrooms = st.slider("Количество спален", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Количество ванных комнат", min_value=1, max_value=10, value=2)
stories = st.slider("Количество этажей", min_value=1, max_value=5, value=2)
parking = st.slider("Количество парковочных мест", min_value=0, max_value=10, value=2)

# Категориальные параметры
mainroad = st.selectbox("Выход на главную дорогу", ["yes", "no"])
guestroom = st.selectbox("Наличие гостевой комнаты", ["yes", "no"])
basement = st.selectbox("Наличие подвала", ["yes", "no"])
hotwaterheating = st.selectbox("Наличие водонагревателя", ["yes", "no"])
airconditioning = st.selectbox("Наличие кондиционера", ["yes", "no"])
prefarea = st.selectbox("Расположение в престижном районе", ["yes", "no"])
furnishingstatus = st.selectbox("Состояние отделки", ["furnished", "semi-furnished", "unfurnished"])

# Кнопка для получения прогноза
if st.button("Рассчитать прогноз стоимости"):
    # Преобразуем вводимые данные
    mainroad_encoded = 1 if mainroad == 'yes' else 0
    guestroom_encoded = 1 if guestroom == 'yes' else 0
    basement_encoded = 1 if basement == 'yes' else 0
    hotwaterheating_encoded = 1 if hotwaterheating == 'yes' else 0
    airconditioning_encoded = 1 if airconditioning == 'yes' else 0
    prefarea_encoded = 1 if prefarea == 'yes' else 0
    furnishingstatus_encoded = label_encoder.transform([furnishingstatus])[0]
    
    # Создаем массив признаков
    features = np.array([[area, bedrooms, bathrooms, stories, mainroad_encoded, guestroom_encoded,
                         basement_encoded, hotwaterheating_encoded, airconditioning_encoded,
                         parking, prefarea_encoded, furnishingstatus_encoded]])
    
    # Делаем предсказание
    prediction = model.predict(features)
    
    # Выводим результат
    st.subheader("Прогноз стоимости недвижимости")
    st.write(f"Предсказанная цена: {prediction[0]:,.0f} рублей")
    
    # Выводим введенные параметры
    st.subheader("Введенные параметры")
    params_df = pd.DataFrame({
        'Параметр': ['Площадь', 'Спален', 'Ванных комнат', 'Этажей', 'Парковочных мест',
                     'Главная дорога', 'Гостевая комната', 'Подвал', 'Водонагреватель',
                     'Кондиционер', 'Престижный район', 'Отделка'],
        'Значение': [area, bedrooms, bathrooms, stories, parking,
                     mainroad, guestroom, basement, hotwaterheating,
                     airconditioning, prefarea, furnishingstatus]
    })
    st.table(params_df)
