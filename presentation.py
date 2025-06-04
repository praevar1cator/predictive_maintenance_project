import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
# Прогнозирование отказов оборудования
---
## 1. Введение
- Цель проекта: создать модель для предсказания отказа оборудования.
- Данные: взяты из UCI Machine Learning Repository (id=601).
- Классификация: бинарная (1 – отказ, 0 – нормальная работа).
---
## 2. Описание данных
- Источник: AI-based predictive maintenance dataset.
- Кол-во признаков: 6 (тип, температура, скорость, крутящий момент, износ и др.).
- Целевая переменная: `Machine failure`.
---
## 3. Этапы проекта
1. Загрузка и очистка данных.
2. Кодирование категориальных признаков.
3. Разделение выборки.
4. Обучение модели (RandomForest).
5. Оценка качества и визуализация.
---
## 4. Предобработка данных
- Удалены лишние столбцы: `UDI`, `Product ID`, TWF, HDF, и т.п.
- Преобразован категориальный признак `Type`.
- Разделение на `X` и `y`, затем — train/test.
---
## 5. Обучение модели
- Использована модель `RandomForestClassifier`.
- Кол-во деревьев: 100, random_state = 42.
- Обучение на 80% данных.
---
## 6. Оценка модели
- Accuracy: 0.96 (примерно).
- Матрица ошибок: визуализирована через `Seaborn`.
- Метрики: `confusion_matrix`, `classification_report`, `roc_auc_score`.
---
## 7. Пример предсказания
- Ввод пользователем параметров оборудования.
- Преобразование типа.
- Вывод результата: отказ или нет, вероятность отказа.
---
## 8. Streamlit-приложение
- Главная страница: анализ + модель.
- Ввод параметров — предсказание результата.
- Отдельная вкладка — текущая презентация.
---
## 9. Заключение
- Модель даёт высокую точность предсказаний.
- Возможные улучшения:
  - Использование других моделей (XGBoost, SVM).
  - Расширение признаков.
  - Учёт временных рядов.
"""

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

if __name__ == "__main__":
    presentation_page()