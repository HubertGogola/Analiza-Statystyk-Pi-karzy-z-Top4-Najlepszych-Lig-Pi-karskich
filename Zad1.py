# Importowanie niezbędnych bibliotek
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Program prównuje statystyki 3 najlepszych strzelców sezonu 2023/2024 z 4 obecnie najlepszych lig piłkarskich na świecie kolejno(Premier League, Serie A, LaLiga, Bundesliga)
# Dane wedlug portalu transfermarkt:

# Ranking najlepszych lig: https://www.transfermarkt.pl/uefa/5jahreswertung/statistik/stat/plus/0?saison_id=2023
# Lista strzelców Premier League: https://www.transfermarkt.pl/premier-league/ewigetorschuetzenliste/pokalwettbewerb/GB1/plus/0/galerie/0?saisonIdVon=2023&saisonIdBis=2023&land_id=
# Lista strzelców Serie A: https://www.transfermarkt.pl/jumplist/torschuetzenliste/wettbewerb/IT1
# Lista strzelców LaLiga: https://www.transfermarkt.pl/jumplist/torschuetzenliste/wettbewerb/ES1
# Lista strzelców Bundesliga: https://www.transfermarkt.pl/jumplist/torschuetzenliste/wettbewerb/L1

# Dane o piłkarzach
data = {
    'Imie': ['Erling', 'Cole', 'Alexander', 'Lautaro', 'Dusan', 'Oliver', 'Artem', 'Alexander', 'Robert', 'Harry', 'Serhou', 'Lois'],
    'Nazwisko': ['Haaland', 'Palmer', 'Isak', 'Martinez', 'Vlahovic', 'Giroud', 'Dovbyk', 'Sorloth', 'Lewandowski', 'Kane', 'Guirassy', 'Openda'],
    'Narodowosc': ['Norwegia', 'Anglia', 'Szwecja', 'Argentyna', 'Serbia', 'Francja', 'Ukraina', 'Norwegia', 'Polska', 'Anglia', 'Francja', 'Belgia'],
    'Klub': ['Manchester City', 'Chelsea', 'Newcastle', 'Inter Mediolan', 'Juventus', 'AC Milan', 'Girona', 'Villareal', 'FC Barcelona', 'Bayern', 'Vfb Stuttgard', 'RB Lipsk'],
    'Wiek': [23, 22, 24, 26, 24, 37, 26, 28, 35, 30, 28, 24],
    'Ilosc_meczy': [31, 34, 30, 33, 33, 35, 36, 34, 35, 32, 28, 34],
    'Bramki': [27, 22, 21, 24, 16, 15, 24, 23, 19, 36, 28, 24]
}

# Tworzenie ramki danych
df = pd.DataFrame(data)

# Wyświetlenie wszystkich piłkarzy z pełnymi danymi
print("Dane piłkarzy:")
for index, row in df.iterrows():
    print(f"Piłkarz {index + 1}:")
    print(f"Imię: {row['Imie']}")
    print(f"Nazwisko: {row['Nazwisko']}")
    print(f"Narodowość: {row['Narodowosc']}")
    print(f"Klub: {row['Klub']}")
    print(f"Wiek: {row['Wiek']}")
    print(f"Ilość meczów: {row['Ilosc_meczy']}")
    print(f"Bramki: {row['Bramki']}")
    print("--------------------")

# Przeprowadzenie analizy danych

# Wizualizacja zależności między zmiennymi
sns.pairplot(df[['Wiek', 'Ilosc_meczy', 'Bramki']])
plt.show()

# Statystyki opisowe
desc_stats = df.describe().round(2)
print("Statystyki opisowe:")
print(desc_stats)

# Korelacja między zmiennymi
corr_matrix = df[['Wiek', 'Ilosc_meczy', 'Bramki']].corr().round(2)
print("Macierz korelacji:")
print(corr_matrix)

# Wizualizacja macierzy korelacji
plt.figure(figsize=(8, 6))
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.show()

# Podział danych na zbiór treningowy i testowy
X = df[['Wiek', 'Ilosc_meczy']]
y = df['Bramki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Budowa prostego modelu liniowego (jedna zmienna niezależna)
simple_model = LinearRegression()
simple_model.fit(X_train[['Wiek']], y_train)
y_pred_simple = simple_model.predict(X_test[['Wiek']])
print(f"Simple Model RMSE: {mean_squared_error(y_test, y_pred_simple, squared=False)}")
print(f"Simple Model R^2: {r2_score(y_test, y_pred_simple)}")

# Budowa modelu liniowego (wszystkie zmienne niezależne)
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
y_pred_multiple = multiple_model.predict(X_test)
print(f"Multiple Model RMSE: {mean_squared_error(y_test, y_pred_multiple, squared=False)}")
print(f"Multiple Model R^2: {r2_score(y_test, y_pred_multiple)}")

# Porównanie modeli
models_comparison = pd.DataFrame({
    "Model": ["Simple (Wiek)", "Multiple (Wiek + Ilosc_meczy)"],
    "RMSE": [mean_squared_error(y_test, y_pred_simple, squared=False), 
             mean_squared_error(y_test, y_pred_multiple, squared=False)],
    "R^2": [r2_score(y_test, y_pred_simple), r2_score(y_test, y_pred_multiple)]
})

print("Porównanie modeli:")
print(models_comparison)