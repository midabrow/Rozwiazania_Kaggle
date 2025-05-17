# ============================================================
# 🔹 IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.colors as colors
from plotly.subplots import make_subplots
from matplotlib import colors
import warnings
from ydata_profiling import ProfileReport
import missingno as msno
from termcolor import colored
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from highlight_text import fig_text
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.classifier import ConfusionMatrix, ROCAUC, PrecisionRecallCurve
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.style import set_palette

# ============================================================
# 🔹 COLORING TERMINAL OUTPUT
# ============================================================
class clr:
    bold = '\033[1m'             
    orange = '\033[38;5;208m'  
    blue = '\033[38;5;75m'    
    reset = '\033[0m' 

    @staticmethod
    def print_colored_text():
        print(clr.orange + "This is dark orange text!" + clr.reset)
        print(clr.blue + "This is light blue text!" + clr.reset)
        print(clr.bold + "This is bold text!" + clr.reset)
        print(clr.bold + clr.orange + "This is bold dark orange text!" + clr.reset)


# ============================================================
# 🔹 UNIQUE VALUES FUNCTIONS
# ============================================================

def check_unique_values(dataset, dataset_name):
    print(clr.orange + f"\n{'='*50}")
    print(f"🔍 Unique Values Report for: {clr.bold + dataset_name}")
    print(f"{'='*50}" + clr.reset)

    unique_values = dataset.apply(lambda x: x.nunique())
    print("\nNumber of unique values per column:")
    print(unique_values)

    object_columns = dataset.select_dtypes(include=['object']).columns
    if len(object_columns) == 0:
        print(colored('❌ There is no object data types in datase', 'red'))
    else:
        for col in object_columns:
            print(f"\nUnique values in column '{col}':")
            print(dataset[col].unique())



# ============================================================
# 🔹 MISSING VALUES FUNCTIONS
# ============================================================
def plot_missing_values(dataset, dataset_name):
    nan_percent = dataset.isnull().mean() * 100

    # Filtering features with missing value
    nan_percent= nan_percent[nan_percent>0].sort_values()
    nan_percent = round(nan_percent,1)

    # Plot the barh chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(nan_percent.index, nan_percent.values, color='#1E3A8A', height=0.65)

    # Annotate the values and indexes
    for i, (value, name) in enumerate(zip(nan_percent.values, nan_percent.index)):
        ax.text(value+0.5, i, f"{value}%", ha='left', va='center', fontweight='bold', color='#1E3A8A', fontsize=12)
    
    # Set x-axis limit
    ax.set_xlim([0,110])

    # Add title and xlabel
    plt.title("Percentage of Missing Data in " + dataset_name, fontsize=14)
    plt.xlabel('Percentages (%)', fontsize=12)
    plt.show()


def report_missing_data(dataset, dataset_name):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = dataset.isnull().mean() * 100
    
    result = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(clr.orange + f"🔴 Missing Data Report for: {dataset_name}")
    print(f"{'-'*50}" + clr.reset)

    print(result[result['Percent'] > 0])

    if dataset.isnull().sum().sum() != 0:
        print(f"\n⚠️ Missing Data Matrix for: {dataset_name}")
        plt.figure(figsize=(10, 5))  
        msno.matrix(dataset)
        plt.title(f"Missing Data in {dataset_name}", fontsize=14) 
        plt.show()

        print(clr.orange + f"\n{'='*50}")
        print(f"🔍 Report for: {clr.bold + dataset_name}")
        print(f"{'='*50}" + clr.reset)
        plot_missing_values(dataset, dataset_name)

def show_percent_of_missing_values(dateset):
    missing_vals = round(dateset.isna().mean() * 100, 1)
    print(clr.orange + "Columns with missing values:" + clr.reset)
    print(missing_vals[missing_vals > 0])
    return 

# ============================================================
# 🔹 NUMERICAL COLUMNS
# ============================================================
import matplotlib.ticker as ticker

def show_dtypes_cols(dataFrame, dtype, datasetType='Train'):
    cols = dataFrame.select_dtypes(include=dtype).columns.to_list()
    print(clr.orange + f'.: {datasetType} Dataset - Numerical Columns :.' + clr.reset)
    print(cols)
    return cols

def plot_numerical_distributions(df, numerical_cols, title='Distribution of Numerical Variables'):
    """
    Generates histograms for numerical columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        title (str): Title for the overall plot.
    """
    n_cols = 2
    n_rows = int(np.ceil(len(numerical_cols) / n_cols))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4 * n_rows))
    ax = ax.flatten()  # Flatten the 2D array of axes for easy iteration
    
    for i, col in enumerate(numerical_cols):
        # Determine bin edges
        if df[col].dtype == 'int64':
            bin_edges = np.arange(df[col].min(), df[col].max() + 1) - 0.5
        else:
            bin_edges = np.histogram_bin_edges(df[col].dropna(), bins='sturges')  # Sturges for more bins
        
        # Plot histogram
        graph = sns.histplot(data=df, x=col, bins=bin_edges, ax=ax[i], edgecolor='black', color='#1E3A8A', alpha=0.6)
        
        # Customize axis labels
        ax[i].set_xlabel(col, fontsize=12)
        ax[i].set_ylabel('Count', fontsize=10)
        ax[i].grid(color='lightgrey', linestyle='--', linewidth=0.5)
        
        # Format x-axis
        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Ensure more values are shown
        ax[i].tick_params(axis='x', rotation=30)
        
        # Add annotations for bar heights
        for p in graph.patches:
            ax[i].annotate(f'{p.get_height():.0f}', 
                           (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                           ha='center', fontsize=8, fontweight="bold")
        
        # Add mean and standard deviation text
        textstr = '\n'.join((
            r'$\mu=%.2f$' % df[col].mean(),
            r'$\sigma=%.2f$' % df[col].std()
        ))
        ax[i].text(0.75, 0.9, textstr, transform=ax[i].transAxes, fontsize=10, verticalalignment='top',
                   color='white', bbox=dict(boxstyle='round', facecolor='#1E3A8A', edgecolor='white', pad=0.5))
    
    # General title and layout
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================
# 🔹 CATEGORICAL COLUMNS
# ============================================================
def plot_categorical_pie(df: pd.DataFrame, categorical_columns: list):
    """
    Generates pie charts for categorical variables in a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    categorical_columns (list): List of categorical column names.
    """
    # Create subplots
    fig = make_subplots(rows=1, cols=len(categorical_columns), specs=[[{'type':'domain'}]*len(categorical_columns)],
                        vertical_spacing=0.01, horizontal_spacing=0.01)
    
    for i, feature in enumerate(categorical_columns):
        value_counts = df[feature].value_counts()
        labels = value_counts.index.tolist()
        values = value_counts.values.tolist()

        # Define color map based on purple
        cmap = colors.LinearSegmentedColormap.from_list("Purple", ["Purple", "white"])
        norm = colors.Normalize(vmin=0, vmax=len(labels))
        color_list = [colors.rgb2hex(cmap(norm(i))) for i in range(len(labels))]

        # Create the pie chart
        pie_chart = go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=color_list, line=dict(color='white', width=3)),
            textposition='inside',
            textinfo='percent+label',
            title=feature,
            title_font=dict(size=25, color='black', family='Calibri')
        )

        # Add the pie chart to the subplot
        fig.add_trace(pie_chart, row=1, col=i+1)

    # Update the layout
    fig.update_layout(showlegend=False, height=400, width=990, 
                      title={
                          'text': "Distribution of Categorical Variables",
                          'y': 0.90,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': {'size': 28, 'color': 'black', 'family': 'Calibri'}
                      })
    
    # Show the plot
    fig.show()


# ============================================================
# 🔹 DUPLICATE VALUES FUNCTIONS
# ============================================================
def check_duplicates(dataset, dataset_name):
    print(clr.orange + f"\n{'='*50}")
    print(f"🔍 Report for: {clr.bold + dataset_name}")
    print(f"{'='*50}" + clr.reset)
    
    total_duplicates = dataset.duplicated().sum()  # Liczba zduplikowanych wierszy
    print(clr.bold + clr.blue + f"\nTotal number of duplicate rows: {total_duplicates}" + clr.reset)
    
    if total_duplicates > 0:
        print("\nDuplicate rows in the dataset:")
        print(dataset[dataset.duplicated()])
    
    print(f"\nChecking duplicates in each column:")
    for col in dataset.columns:
        duplicate_values = dataset[col].duplicated().sum()
        print(f"Column '{clr.bold + col + clr.reset}' has {duplicate_values} duplicate values")
    
    print(f"{'='*50}\n")



# ============================================================
# 🔹 CLEANING DATA
# ============================================================
def clean_column_names(dataset: pd.DataFrame, dataset_name: str, to_lower: bool = True, 
                        replace_chars: bool = False, old_char: str = None, new_char: str = None, 
                        remove_special_chars: bool = False):
    """
    Cleans column names in a dataset by applying transformations such as:
    - Stripping leading/trailing spaces
    - Converting to lowercase (optional)
    - Replacing specified characters (optional)
    - Removing special characters (optional)
    - Checking for duplicate column names

    Parameters:
    - dataset (pd.DataFrame): The dataset whose column names need to be cleaned.
    - dataset_name (str): Name of the dataset (for display purposes).
    - to_lower (bool): If True, converts all column names to lowercase (default is True).
    - replace_chars (bool): If True, replaces occurrences of old_char with new_char in column names (default is False).
    - old_char (str): Character(s) to be replaced when replace_chars is True.
    - new_char (str): Character(s) to replace old_char when replace_chars is True.
    - remove_special_chars (bool): If True, removes non-alphanumeric characters from column names (default is True).

    Returns:
    - None (modifies dataset in place)
    """
    print(clr.orange + f"\n{'='*50}")
    print(f"🔍 Checking and cleaning column names for: {clr.bold}{dataset_name}")
    print(f"{'='*50}" + clr.reset)
    
    original_columns = dataset.columns.tolist()
    print(f"\nOriginal column names: {original_columns}")
    
    cleaned_columns = []
    for col in original_columns:
        col = col.strip()  # Remove leading/trailing spaces
        
        if replace_chars and old_char and new_char:
            col = col.replace(old_char, new_char)  # Replace specified character
        
        if to_lower:
            col = col.lower()  # Convert to lowercase
        
        if remove_special_chars:
            col = ''.join(e if e.isalnum() or e == '_' else '' for e in col)  # Remove special characters
        
        cleaned_columns.append(col)
    
    dataset.columns = cleaned_columns
    print(f"\nCleaned column names: {dataset.columns.tolist()}")
    
    # Check for duplicate column names
    duplicates = dataset.columns[dataset.columns.duplicated()].tolist()
    if duplicates:
        print(f"\n⚠️ Duplicate columns found: {duplicates}")
    else:
        print("\n✅ No duplicate columns found.")
    



# ============================================================
# 🔹 OUTLIERS
# ============================================================
from scipy.stats import zscore


def check_outliers(dataset, dataset_name, numerical_cols=None, target_col=None):
    print(clr.orange + f"\n{'='*50}")
    print(f"🔍 Outliers Report for: {clr.bold + dataset_name}")
    print(f"{'='*50}" + clr.reset)

    # Wybór kolumn numerycznych, jeśli nie podano
    if numerical_cols is None:
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns

    num_rows = len(numerical_cols)  # Liczba wierszy w siatce wykresów
    fig, ax = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))  
    ax = ax.reshape(num_rows, 2)  # Upewniamy się, że mamy poprawny układ tablicy osi

    for i, col in enumerate(numerical_cols):
        print(f"\n📊 Checking outliers in column:" + clr.orange + clr.bold + f"{col}" + clr.reset)

        # 1️⃣ Boxplot (po lewej stronie)
        sns.boxplot(x=dataset[col], color='#6f80b3', ax=ax[i, 0])
        ax[i, 0].set_title(f"Boxplot of {col}")

        # 2️⃣ Scatterplot (po prawej stronie, jeśli podano target_col)
        if target_col:
            sns.scatterplot(x=dataset[col], y=dataset[target_col], hue=dataset[target_col], palette="coolwarm", ax=ax[i, 1])
            ax[i, 1].set_title(f"Scatterplot of {col} vs {target_col}")
        else:
            ax[i, 1].axis("off")  # Jeśli nie ma target_col, wyłączamy ten subplot

        # 3️⃣ IQR – Wartości odstające
        Q1, Q3 = dataset[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_iqr = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
        print(clr.blue + f"📌 Outliers based on IQR (<= {lower_bound:.2f} or >= {upper_bound:.2f}):" + clr.reset + f"{len(outliers_iqr)} values")
        print(outliers_iqr[[col]].head())

        # 4️⃣ Z-score – Wartości odstające
        z_scores = zscore(dataset[col].fillna(0))  # Obsługa NaN
        outliers_zscore = dataset[np.abs(z_scores) > 3]
        print(clr.blue + f"📌 Outliers based on Z-score (>|3|):" + clr.reset + f"{len(outliers_zscore)} values")
        print(outliers_zscore[[col]].head())

    plt.tight_layout()
    plt.show()


# ============================================================
# 🔹 BALANCE
# ============================================================

def check_balance_in_data(df, column):
    return round(df[column].value_counts(normalize=True) * 100, 0)


def plot_target_balance(df, target_column, chart_type="bar"):
    """
    Funkcja do wizualizacji balansu targetu.
    
    :param df: DataFrame zawierający dane
    :param target_column: Nazwa kolumny targetu
    :param chart_type: Rodzaj wykresu - "bar" (domyślnie) lub "pie"
    """
    # Zliczanie wartości unikalnych
    target_counts = df[target_column].value_counts().sort_index()
    
    # Kolory dla lepszej czytelności
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

    if chart_type == "bar":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=target_counts.index.astype(str), 
            y=target_counts.values, 
            marker_color=colors[:len(target_counts)],
            text=target_counts.values, 
            textposition='auto'
        ))
        fig.update_layout(title=f"Balans wartości {target_column}",
                          xaxis_title=target_column,
                          yaxis_title="Liczność",
                          template="plotly_dark")

    elif chart_type == "pie":
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=target_counts.index.astype(str), 
            values=target_counts.values,
            marker=dict(colors=colors[:len(target_counts)]),
            textinfo="label+percent"
        ))
        fig.update_layout(title=f"Procentowy udział {target_column}", template="plotly_dark")

    else:
        raise ValueError("chart_type powinien być 'bar' lub 'pie'")

    fig.show()



# ============================================================
# 🔹 BUILDING MODEL
# ============================================================

def metrics_calculator(y_pred, y_test, model_name):
    if y_pred is None or model_name is None:
        raise ValueError("Prediction or model name is None. Check your input values.")
    
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns=[str(model_name)])

    print("Kolumny w metrics_results:", result.columns)
    print("Indeksy w metrics_results:", result.index)

    
    result = (result * 100).round(2).astype(str) + '%'                            
    return result

def plot_classification_report(y_true, y_pred, ax, set_name='training set') :

    # Generowanie raportu klasyfikacji jako DataFrame
    cr = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    
    # Rysowanie heatmapy na podanej osi
    sns.heatmap(cr, cmap='Purples', annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    
    # Dostosowanie tytułu
    ax.set_title(f'Classification Report ({set_name})', fontsize=12, fontweight='bold')
    
    # Ustawienie etykiet poziomych na górze
    ax.xaxis.tick_top()

def plot_confusion_matrix(model, X_train, y_train, X_test, y_test, ax, title_style, xy_label):
    labels = np.unique(y_test).astype(str)
    conf_matrix = ConfusionMatrix(model, ax=ax, cmap='Blues')
    conf_matrix.fit(X_train, y_train)
    conf_matrix.score(X_test, y_test)
    conf_matrix.finalize()
    conf_matrix.ax.set_title('Confusion Matrix\n', **title_style)
    conf_matrix.ax.set_xlabel('\nPredicted Class', **xy_label)
    conf_matrix.ax.set_ylabel('True Class\n', **xy_label)
    conf_matrix.ax.xaxis.set_ticklabels(labels, rotation=0)
    conf_matrix.ax.yaxis.set_ticklabels(labels)

def plot_roc_auc(model, X_train, y_train, X_test, y_test, ax, title_style, xy_label, tick_params, grid_style, binary=False):
    logrocauc = ROCAUC(model, classes=['False', 'True'] if binary else None, ax=ax, binary=binary)
    logrocauc.fit(X_train, y_train)
    logrocauc.score(X_test, np.array(y_test))
    logrocauc.finalize()
    logrocauc.ax.set_title('ROC AUC Curve\n', **title_style)
    logrocauc.ax.set_xlabel('\nFalse Positive Rate', **xy_label)
    logrocauc.ax.set_ylabel('True Positive Rate\n', **xy_label)
    logrocauc.ax.tick_params(axis='both', labelsize=10, bottom='on', left='on', **tick_params)
    logrocauc.ax.grid(axis='both', alpha=0.4, **grid_style)

def plot_learning_curve(best_model, X_train, y_train, ax, title_style, xy_label, tick_params, grid_style):
    lcurve = LearningCurve(best_model, ax=ax, colors='yellowgreen')
    lcurve.fit(X_train, y_train)
    lcurve.finalize()
    lcurve.ax.set_title('Learning Curve\n', **title_style)
    lcurve.ax.set_xlabel('\nTraining Instances', **xy_label)
    lcurve.ax.set_ylabel('Scores\n', **xy_label)
    lcurve.ax.tick_params(axis='both', labelsize=10, bottom='on', left='on', **tick_params)
    lcurve.ax.grid(axis='both', alpha=0.4, **grid_style)
    for spine in lcurve.ax.spines.values(): 
        spine.set_color('None')

def plot_feature_importances_or_precision_recall(best_model, X_train, y_train, X_test, y_test, ax, title_style, xy_label, tick_params, grid_style):
    labels = np.unique(y_test).astype(str)
    try:
        feat_importance = FeatureImportances(best_model, ax=ax, labels=labels, topn=5, colors='yellowgreen')
        feat_importance.fit(X_train, y_train)
        feat_importance.finalize()
        feat_importance.ax.set_title('Feature Importances (Top 5 Features)\n', **title_style)
        feat_importance.ax.set_xlabel('\nRelative Importance', **xy_label)
        feat_importance.ax.set_ylabel('Features\n', **xy_label)
        feat_importance.ax.grid(axis='x', alpha=0.4, **grid_style)
        feat_importance.ax.grid(axis='y', alpha=0, **grid_style)
        for spine in feat_importance.ax.spines.values(): 
            spine.set_color('None')
    except:
        # Precision-Recall Curve if Feature Importances are unavailable
        prec_curve = PrecisionRecallCurve(best_model, ax=ax, ap_score=True, iso_f1_curves=True)
        prec_curve.fit(X_train, y_train)
        prec_curve.score(X_test, y_test)
        prec_curve.finalize()
        prec_curve.ax.set_title('Precision-Recall Curve\n', **title_style)
        prec_curve.ax.tick_params(axis='both', labelsize=10, bottom='on', left='on', **tick_params)
        prec_curve.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, borderpad=2, frameon=False, fontsize=10)
        for spine in prec_curve.ax.spines.values(): 
            spine.set_color('None')
        prec_curve.ax.set_xlabel('\nRecall', **xy_label)
        prec_curve.ax.set_ylabel('Precision\n', **xy_label)




def plot_performance_summary(y_pred, y_test, model_class, ax, title_style):

    result = metrics_calculator(y_pred, y_test, model_class.__name__)

    # Tworzenie tabeli
    table = ax.table(cellText=result.values, colLabels=result.columns, rowLabels=result.index, loc='best')
    table.scale(0.6, 3.6)
    table.set_fontsize(12)

    # Ukrywanie osi dla obszaru tabeli
    ax.axis('tight')
    ax.axis('off')

    # Tytuł tabeli
    ax.set_title('Performance Summary on Test Data\n', **title_style)

    # Zmiana koloru wiersza nagłówka tabeli
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Nagłówek tabeli
            cell.set_color('#1e3a8a')
            cell.get_text().set_color('white')



def train_and_evaluate(model_class, params, X_train, y_train, X_test, y_test=None, n_splits=3, problem_type='regression', plot_metrics=False, binary=True):


    if problem_type == 'regression':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            model = model_class(**params)
            model.fit(X_train[train_idx], y_train.iloc[train_idx])
            predictions[val_idx] = model.predict(X_train[val_idx])
            test_predictions += model.predict(X_test) / n_splits
        overall_rmse = np.sqrt(mean_squared_error(y_train, predictions))
        return overall_rmse, test_predictions

    elif problem_type == 'classification':
        # --- Apply Grid Search ---
        model = GridSearchCV(model_class(), param_grid=params, cv=n_splits, n_jobs=-1, scoring='accuracy', verbose=1)

        # --- Fitting Model ---
        fit_model = model.fit(X_train, y_train)
        print(fit_model)

        # --- Best Estimators ---
        best_model = model.best_estimator_
        print(f"\n{clr.orange}Best Model:{clr.reset} {best_model}")

        # --- Prediction Results ---
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # --- Accuracy & Best Score ---
        best_score = model.best_score_
        acc_score_train = accuracy_score(y_train, y_pred_train)
        acc_score = accuracy_score(y_test, y_pred_test) if y_test is not None else None

        print(f'\n{clr.orange}Train Accuracy:{clr.reset} {acc_score_train * 100:.2f}%')
        if y_test is not None:
            print(f'{clr.orange}Test Accuracy:{clr.reset} {acc_score * 100:.2f}%')
        print(f'{clr.orange}Best Cross-Validation Score:{clr.reset} {best_score * 100:.2f}%')

        # --- Figures Settings ---
        xy_label = dict(fontweight='bold', fontsize=12)
        grid_style = dict(color='gray', linestyle='dotted', zorder=1)
        title_style = dict(fontsize=14, fontweight='bold')
        tick_params = dict(length=3, width=1, color='black')

        # --- Visualization ---
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, _)) = plt.subplots(4, 2, figsize=(18, 15))
        
        
        plot_classification_report(y_train, y_pred_train, ax1, set_name='training set')
        plot_classification_report(y_test, y_pred_test, ax2, set_name='test set')

        # --- Confusion Matrix ---
        plot_confusion_matrix(best_model, X_train, y_train, X_test, y_test, ax3, title_style, xy_label)

        # --- ROC AUC ---
        plot_roc_auc(best_model, X_train, y_train, X_test, y_test, ax4, title_style, xy_label, tick_params, grid_style, binary=False)

        # --- Learning Curve ---
        plot_learning_curve(best_model, X_train, y_train, ax5, title_style, xy_label, tick_params, grid_style)

        # --- Feature Importance ---
        plot_feature_importances_or_precision_recall(best_model, X_train, y_train, X_test, y_test, ax6, title_style, xy_label, tick_params, grid_style)
        
        # --- Metrics Summary ---
        plot_performance_summary(y_pred_test, y_test, model_class, ax7, title_style) 
        
        plt.tight_layout()

        metrics_results = metrics_calculator(y_pred_test, y_test, model_class.__name__)
        return best_model, acc_score_train, acc_score, metrics_results
    



def evaluate_models(models, X_train, y_train, X_test, y_test, n_splits=5, scale=False, problem_type='classification'):
    """
    Funkcja testuje różne modele klasyfikacyjne, przeprowadza GridSearchCV, trenuje najlepszy model 
    i zapisuje wyniki jego działania.
    
    Parametry:
    - models (list): Lista słowników zawierających modele, ich nazwy i hiperparametry.
    - X_train, y_train: Dane treningowe.
    - X_test, y_test: Dane testowe.
    - n_splits (int): Liczba podziałów dla walidacji krzyżowej.
    - scale (bool): Czy skalować dane? (opcjonalnie)
    - problem_type (str): Typ problemu ('classification' lub 'regression').
    
    Zwraca:
    - results (list): Lista słowników z wynikami modeli.
    """
    results = []

    for model_info in models:
        model_name = model_info['name']
        model_class = model_info['model']
        params = model_info['params']
        
        print(f"\n{clr.orange}{'='*20} Model testing: {model_name} {'='*20}{clr.reset}\n")

        # Trenowanie modelu z najlepszymi hiperparametrami
        best_model, train_score, test_score, metrics_results = train_and_evaluate(
            model_class, params, X_train, y_train, X_test, y_test, 
            n_splits=n_splits, problem_type=problem_type
        )

        results.append({
            'model': model_name,
            'train_score': train_score,
            'test_score': test_score,
            'best_params': best_model.get_params(),
            'precision': metrics_results.loc['Precision', model_class.__name__],
            'recall': metrics_results.loc['Recall', model_class.__name__],
            'F1-score': metrics_results.loc['F1-score', model_class.__name__]
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model').T
    return df_results
