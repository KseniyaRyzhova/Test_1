import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

# Заголовок приложения
st.title("Прогноз стоимости недвижимости")
st.write("Выберите район и введите площадь")

# Функция для создания и обучениямодели
@st.cache_resource  # Кэшируем модель, чтобы не пересоздавать при каждом запуске
def create_simple_model():
    # данные: площадь, район (1-престижный, 0-обычный), цена
    data = {
        'area': [5000, 6000, 7000, 8000, 9000, 10000,
                 4000, 5000, 6000, 7000, 8000, 9000],
        'location': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'price': [12000000, 13000000, 14000000, 15000000, 16000000, 17000000,
                  6000000, 7000000, 8000000, 9000000, 10000000, 11000000]
    }
    df = pd.DataFrame(data)
    
    X = df[['area', 'location']]
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Загружаем или создаем модель
model = create_simple_model()

# Элементы интерфейса
area = st.slider("Площадь (кв. метров)", 1000, 15000, 6000)
location = st.selectbox("Район", ["Обычный", "Престижный"])

# Преобразуем выбор в число
location_encoded = 1 if location == "Престижный" else 0

# Кнопка для расчета
if st.button("Рассчитать цену"):
    # Делаем предсказание
    features = np.array([[area, location_encoded]])
    prediction = model.predict(features)[0]
    
    # Выводим результат
    st.success(f"Прогнозируемая цена: {prediction:,.0f} рублей")
    
    # Показываем введенные данные
    st.write(f"Площадь: {area} кв. метров")
    st.write(f"Район: {location}")
