import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from model.train_model import train_and_save_model

def load_data():
    data = pd.read_csv('/data/wine_data.csv')
    return data

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Train the model first.")
        return None

def show_dataframe(df):
    st.write(df.head())

def plot_distribution(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

def plot_comparison(df, col1, col2):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
    st.pyplot(fig)

st.title('Wine Recognition App')
df = load_data()

if st.sidebar.button('Show Dataset Description'):
    st.header("Opis zbioru danych")
    st.write("""
        Dane dotyczą analizy chemicznej win wyprodukowanych w tym samym regionie we Włoszech,
        ale pochodzących z trzech różnych kultywarów. Analizowano ilości 13 składników znajdujących się
        w każdym z trzech rodzajów win. Dane idealne do pierwszych testów nowego klasyfikatora.
    """)
    st.subheader("Opis zmiennych")
    st.write("""
        1. Wine category (cultivar) - Kategoria wina, zazwyczaj reprezentująca różne rodzaje winogron (kultywary) z których wino zostało wyprodukowane. Jest to zmienna celu w kontekście klasyfikacji win.
        2. Alcohol - Procentowa zawartość alkoholu w winie, co jest standardowym pomiarem dla wszystkich alkoholi.
        3. Malic acid - Ilość kwasu jabłkowego w winie. Kwas ten jest jednym z głównych kwasów występujących w owocach i jest ważnym składnikiem wpływającym na smak wina.
        4. Ash - Całkowita zawartość mineralna (popiół) w winie, mierzona po spaleniu próbki. Ash może dać wskazówkę o glebie i warunkach, w jakich uprawiane były winogrona.
        5. Alcalinity of ash - Miara zasadowości popiołu w winie, co może wpłynąć na ogólny balans kwasowości wina.
        6. Magnesium - Ilość magnezu w winie, co jest jednym z elementów mających wpływ na zdrowotne właściwości wina.
        7. Total phenols - Całkowita zawartość fenoli w winie. Fenole są ważnymi składnikami, które wpływają na smak, kolor i trwałość wina.
        8. Flavanoids - Grupa fenoli specyficzna dla win, które mają silne właściwości antyoksydacyjne i są ważne dla koloru i smaku wina.
        9. Nonflavanoid phenols - Inna grupa związków fenolowych w winie, które również wpływają na jego smak i właściwości antyoksydacyjne, ale są mniej skupione na kolorze.
        10. Proanthocyanins - Grupa fenoli znanych z korzystnego wpływu na zdrowie serca oraz są one ważne dla smaku i koloru wina.
        11. Color intensity - Intensywność koloru wina, która jest bezpośrednio związana z ilością i rodzajem fenoli w winie.
        12. Hue - Odcień wina, który jest kolejnym wskaźnikiem jego charakterystyki, bazując na czym wino może być klasyfikowane na czerwone, białe itd.
        13. OD280/OD315 of diluted wines - Wskaźnik absorpcji wina przy określonych długościach fal światła, używany do określenia zawartości białka w winie.
        14. Proline - Ilość proliny, aminokwasu, który jest obecny w wysokich stężeniach w niektórych winach i może być wskaźnikiem ich jakości oraz pochodzenia.
    """)

model_choice = st.sidebar.selectbox('Choose a model to train:', ['logistic_regression', 'random_forest', 'svm'])

if st.sidebar.button('Train Model'):
    try:
        accuracy, cm = train_and_save_model(model_choice, '/data/wine_data.csv', '/data/wine_model.pkl')
        model = load_model('/data/wine_model.pkl')
        if model:
            st.write('Model trained successfully!')
            st.write(f"Model trained: {model_choice}")
            st.write(f"Accuracy of the model is: {accuracy * 100:.2f}%")
            st.write("Confusion Matrix:")
            st.write(cm)
    except ValueError as e:
        st.error(str(e))
    except Exception as ex:
        st.error(f"An error occurred: {str(ex)}")

if st.sidebar.button('Show Data'):
    show_dataframe(df)

if st.sidebar.button('Show Stats'):
    st.write(df.describe())

variable = st.sidebar.selectbox('Choose variable to plot:', df.columns)
if st.sidebar.button('Plot Distribution'):
    plot_distribution(df, variable)

selected_cols = st.sidebar.multiselect('Choose two variables for comparison:', df.columns, default=list(df.columns[:2]))
if len(selected_cols) == 2:
    col1, col2 = selected_cols
    if st.sidebar.button('Compare Variables'):
        plot_comparison(df, col1, col2)