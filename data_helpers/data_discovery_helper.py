import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from IPython.display import display

class DataInspector:
    def __init__(self, df: pd.DataFrame, column_name: str = None):
        """
        Klasa analizująca DataFrame lub pojedynczą kolumnę.
        
        :param df: Pandas DataFrame do analizy
        :param column_name: (Opcjonalnie) Nazwa kolumny do analizy
        """
        self.df = df
        self.column_name = column_name
        self.is_single_column = column_name is not None

        if self.is_single_column and column_name not in df.columns:
            raise ValueError(f"Kolumna '{column_name}' nie istnieje w DataFrame")

    def basic_info(self):
        """Zwraca podstawowe informacje o DataFrame lub kolumnie."""
        if self.is_single_column:
            print(f"\n📌 Podstawowe informacje o kolumnie: {self.column_name}")
            print(f" 🔸 Liczba wierszy: {self.df[self.column_name].shape[0]}")
            print(f" 🔸 Typ danych: {self.df[self.column_name].dtype}")
            # print(f"Liczba unikalnych wartości: {self.df[self.column_name].nunique()}")
            # print(f"Liczba braków: {self.df[self.column_name].isnull().sum()} ({self.df[self.column_name].isnull().mean() * 100:.2f}%)")
        else:
            print("\n📌 Podstawowe informacje o całym DataFrame:")
            print(f" 🔸 Liczba wierszy: {self.df.shape[0]}")
            print(f" 🔸 Liczba kolumn: {self.df.shape[1]}")
            print(f" 🔸 Info:")
            display(self.df.info())
            print("\n📝 Pierwsze 5 wierszy:")
            display(self.df.head())

    def describe_data(self):
        """Zwraca statystyki dla całego DataFrame lub jednej kolumny."""
        if self.is_single_column:
            print(f"\n📊 Statystyki dla kolumny: {self.column_name}")
            if pd.api.types.is_numeric_dtype(self.df[self.column_name]):
                display(self.df[self.column_name].describe())
            else:
                display(self.df[self.column_name].value_counts().head(10))
        else:
            print("\n📊 Statystyki dla całego DataFrame:")
            display(self.df.describe(include="all"))

    def check_missing_data(self):
        """Analizuje brakujące wartości w DataFrame lub pojedynczej kolumnie."""
        if self.is_single_column:
                percent_missing = self.df[self.column_name].isnull().mean() * 100
                if percent_missing == 0:
                    print(f"\n✅ Brak brakujących wartości w kolumnie {self.column_name}!")
                else:
                    print(f"\n🚨 Braki w kolumnie {self.column_name}: {percent_missing:.2f}%")
        else:
            missing = self.df.isnull().sum()
            total_missing = missing.sum()  # Suma wszystkich braków

            if total_missing == 0:
                print("\n✅ Brak brakujących wartości!")
            else:
                print("\n🚨 Brakujące wartości:")
                missing_percent = (missing / len(self.df)) * 100
                missing_data = pd.DataFrame({'Liczba braków': missing, 'Procent braków': missing_percent.round(2)})

                # Wyświetl tylko kolumny z brakami
                missing_data = missing_data[missing_data["Liczba braków"] > 0].sort_values(by="Liczba braków", ascending=False)
                display(missing_data)

                msno.matrix(self.df)

    def check_duplicates(self):
        """Analiza duplikatów"""
        if self.is_single_column: 
            duplicates = self.df[self.column_name].duplicated().sum()
            print(f"🔍 Liczba duplikatów: {duplicates}")

            if duplicates > 0:
                display(self.df[self.column_name][self.df[self.column_name].duplicated()].head())
        else:
            duplicates = self.df.duplicated().sum()
            print(f"🔍 Liczba duplikatów: {duplicates}")

            if duplicates > 0:
                display(self.df[self.df.duplicated()].head())


    def check_outliers(self):
        """Sprawdza wartości odstające w kolumnach numerycznych."""
        if self.is_single_column:
            if pd.api.types.is_numeric_dtype(self.df[self.column_name]):
                q1 = self.df[self.column_name].quantile(0.25)
                q3 = self.df[self.column_name].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = self.df[(self.df[self.column_name] < lower_bound) | (self.df[self.column_name] > upper_bound)]
                print(f"\n📈 Liczba wartości odstających w {self.column_name}: {outliers.shape[0]}")
            else:
                print(f"\n⚠️ Kolumna {self.column_name} nie jest numeryczna, więc nie można sprawdzić wartości odstających.")
        else:
            num_cols = self.df.select_dtypes(include=np.number).columns
            outlier_counts = {}
            for col in num_cols:
                q1, q3 = self.df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_counts[col] = outliers.shape[0]
            print("\n📈 Liczba wartości odstających w kolumnach numerycznych:")
            display(pd.Series(outlier_counts).to_frame("Liczba wartości odstających"))

        num_df = self.df.select_dtypes(include=["number"])  # Tylko kolumny numeryczne
        melted_df = num_df.melt(var_name="Kolumna", value_name="Wartość")  # Przekształcenie DF do formatu długiego

        fig = px.box(melted_df, x="Kolumna", y="Wartość", title="Interaktywny Boxplot zmiennych liczbowych")
        fig.update_layout(xaxis_tickangle=-90)  # Obrót etykiet dla czytelności
        fig.show()

    def check_data_consistency(self):
        """Sprawdza poprawność wartości liczbowych i dat"""
        numeric_cols = self.df.select_dtypes(include=["number"])
        date_cols = self.df.select_dtypes(include=["datetime"])
        
        for col in numeric_cols:
            if (self.df[col] < 0).any():
                print(f"⚠️ Ostrzeżenie: Kolumna '{col}' zawiera wartości ujemne!")
            
        for col in date_cols:
            if (self.df[col] > pd.Timestamp.today()).any():
                print(f"⚠️ Ostrzeżenie: Kolumna '{col}' zawiera daty w przyszłości!")

    def run_full_analysis(self):
        """Uruchamia pełną analizę na całym DataFrame lub jednej kolumnie."""
        self.basic_info()
        self.describe_data()
        self.check_missing_data()
        self.check_duplicates()
        # self.check_unique_values()
        self.check_outliers()
        self.check_data_consistency()
