import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Анализ данных и модель классификации")
    
    # Загрузка данных
    st.header("1. Загрузка данных")
    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        
        # Проверка наличия столбцов перед удалением
        columns_to_drop = []
        if 'UDI' in data.columns:
            columns_to_drop.append('UDI')
        if 'Product ID' in data.columns:
            columns_to_drop.append('Product ID')
        
        # Удаляем только существующие столбцы
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
        
        # Удаляем дополнительные целевые переменные (если есть)
        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        existing_failure_cols = [col for col in failure_columns if col in data.columns]
        if existing_failure_cols:
            data = data.drop(columns=existing_failure_cols)
            
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        st.stop()
    
    # Показываем информацию о данных
    st.subheader("Первые 5 строк данных")
    st.write(data.head())
    
    # Предобработка
    st.header("2. Предобработка данных")
    if 'Type' in data.columns:
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
    else:
        st.warning("Столбец 'Type' не найден в данных")
    
    # Разделение данных
    st.header("3. Разделение данных")
    if 'Machine failure' not in data.columns:
        st.error("Целевая переменная 'Machine failure' не найдена")
        st.stop()
        
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    st.header("4. Обучение модели")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка модели
    st.header("5. Оценка модели")
    y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
    
    # Предсказания
    st.header("6. Предсказание")
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")
        
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.number_input("Air temperature [K]", value=300.0)
            process_temp = st.number_input("Process temperature [K]", value=310.0)
            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500)
        with col2:
            torque = st.number_input("Torque [Nm]", value=40.0)
            tool_wear = st.number_input("Tool wear [min]", value=0)
            type_ = st.selectbox("Type", options=["L", "M", "H"])
        
        submitted = st.form_submit_button("Предсказать")
        
        if submitted:
            # Преобразование категориального признака
            type_mapping = {"L": 0, "M": 1, "H": 2}
            type_encoded = type_mapping[type_]
            
            input_data = pd.DataFrame([[type_encoded, air_temp, process_temp, 
                                     rotational_speed, torque, tool_wear]], 
                                   columns=X_train.columns)
            
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]
            
            st.success(f"Результат: {'Отказ оборудования' if prediction == 1 else 'Нормальная работа'}")
            st.write(f"Вероятность отказа: {proba:.2%}")

if __name__ == "__main__":
    main()