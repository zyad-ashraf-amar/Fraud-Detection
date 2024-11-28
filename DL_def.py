import random
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import copy
import logging
import missingno
import operator
from collections import Counter
from category_encoders import TargetEncoder, BinaryEncoder
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
    SVMSMOTE,
    BorderlineSMOTE,
    ADASYN,
    SMOTEN,
    SMOTENC
)
from imblearn.under_sampling import (
    TomekLinks, 
    RandomUnderSampler,
    EditedNearestNeighbours, 
    RepeatedEditedNearestNeighbours, 
    AllKNN, 
    CondensedNearestNeighbour, 
    ClusterCentroids, 
    NearMiss
)
from sklearn.model_selection import (
    LeaveOneOut,
    LeavePOut,
    RepeatedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    KFold
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelBinarizer, 
    MultiLabelBinarizer,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer
)
from sklearn.feature_extraction import (
    DictVectorizer, 
    FeatureHasher
)
from sklearn.ensemble import (
    VotingClassifier, 
    VotingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ( # type: ignore
    Input, 
    Dense, 
    Dropout, 
    BatchNormalization, 
    ReLU, 
    LeakyReLU, 
    PReLU, 
    Activation, 
    Conv2D, 
    Flatten, 
    ELU,
    LSTM, 
    GRU
)
from tensorflow.keras.activations import swish # type: ignore
from tensorflow.keras.losses import ( # type: ignore
    MeanSquaredError, 
    MeanAbsoluteError, 
    Huber,
    BinaryCrossentropy, 
    Hinge, 
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy
)
from tensorflow.keras.models import ( # type: ignore
    Model, 
    Sequential 
)
from tensorflow.keras.regularizers import ( # type: ignore
    l2, 
    l1
)
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import ( # type: ignore
    SGD, 
    Adagrad, 
    RMSprop, 
    Adam, 
    Adadelta, 
    Adamax, 
    Nadam
)
from tensorflow.keras.initializers import ( # type: ignore
    GlorotUniform, 
    HeNormal
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from scipy.stats import uniform, randint
from fast_ml.model_development import train_valid_test_split
from typing import (
    Any,
    Literal,
    Union,
    List,
    Optional,
    Dict,
    Tuple
)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)


def read_data(file_path: str, sheet_name: str = None, handle_duplicates: bool = True):
    """
    Read data from a file and return a DataFrame. Supports CSV, TXT, Excel, JSON, and HTML files.
    
    Parameters:
    - file_path: The path to the data file.
    - sheet_name: The name of the sheet to read from an Excel file (default is None).
    - handle_duplicates: Whether to drop duplicate rows (default is True).
    
    Returns:
    - A DataFrame or a list of DataFrames (in case of HTML).
    
    Raises:
    - ValueError: If the file format is not supported.
    """
    
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension in ['csv', 'txt']:
            data = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            if sheet_name is None:
                sheet_name = input('Enter the sheet name: ')
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension == 'json':
            data = pd.read_json(file_path)
        elif file_extension == 'html':
            data = pd.read_html(file_path)
            if len(data) == 1:
                data = data[0]
        else:
            raise ValueError('Unsupported file format.')
        
        # Deep copy the data to avoid modifying the original data
        df = copy.deepcopy(data)
        
        # Handle duplicates if required
        if handle_duplicates:
            duplicated_num = df.duplicated().sum()
            if duplicated_num == 0:
                print('the DataFrame dont have any duplicates row')
            else:
                print(f'the DataFrame have {duplicated_num} duplicates rows')
                df = df.drop_duplicates()
                print('the DataFrame without duplicates rows')
        
        print(f'Data read successfully from {file_path}')
        return df
    
    except Exception as e:
        print(f'Error reading data from {file_path}: {str(e)}')
        raise

def columns_info(df):
    cols=[]
    dtype=[]
    unique_v=[]
    n_unique_v=[]
    number_of_rows = df.shape[0]
    number_of_null = []
    for col in df.columns:
        cols.append(col)
        dtype.append(df[col].dtypes)
        unique_v.append(df[col].unique())
        n_unique_v.append(df[col].nunique())
        number_of_null.append(df[col].isnull().sum())
    
    return pd.DataFrame({'names':cols, 'dtypes':dtype, 'unique':unique_v, 'n_unique':n_unique_v, 'number_of_rows':number_of_rows, 'number_of_null':number_of_null}) 


def target_last_col(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Moves the target column to the last position in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The name of the column to move to the last position.

    Returns:
    pd.DataFrame: A DataFrame with the target column moved to the last position.
    
    Raises:
    KeyError: If the target column is not found in the DataFrame.
    """
    if target not in df.columns:
        raise KeyError(f"Column '{target}' not found in DataFrame.")
    
    # Copy the target column
    target_column = df[target].copy()
    
    # Drop the original target column
    df.drop(columns=[target], inplace=True)
    
    # Insert the target column at the end
    df[target] = target_column
    
    return df


def not_useful_columns(df, column_name):
    # Convert single column name to list
    if isinstance(column_name, str):
        column_name = [column_name]
    # Check for each column in the list
    for col in column_name:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    # Drop the columns
    df.drop(column_name, axis=1, inplace=True)


def remove_missing_rows(df, column_name=None):
    if column_name is not None and isinstance(column_name, str):
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")
        df.dropna(subset=[column_name], inplace=True)
    elif column_name is not None and isinstance(column_name, list):
        for col in column_name:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        df.dropna(subset=column_name, inplace=True)
    elif column_name is None:
        df.dropna(inplace=True)


def convert_to_numeric(df, column_name):
    # Convert single column name to list
    if isinstance(column_name, str):
        column_name = [column_name]
    
    # Check for each column in the list
    for col in column_name:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    
    # Apply pd.to_numeric to each column
    for col in column_name:
        df[col] = pd.to_numeric(df[col], errors='coerce')



def fill_missing_values_dataFrame(df: pd.DataFrame, 
                         model: Literal['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill'] = 'KNNImputer', 
                         n_neighbors: int = 5, 
                         weights: str = 'uniform', 
                         strategy: str = 'mean', 
                         fill_value = None,
                         estimator = None,
                         max_iter = 10,
                         tol = 0.001,
                         constant: Union[int, float] = 0
                         ) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame using the specified imputation strategy.
    
    Parameters:
    df (pd.DataFrame): DataFrame with missing values.
    model (str): Imputation strategy to use ('KNNImputer', 'SimpleImputer', 'constant', 
                'mean', 'median', 'mode', 'interpolation', 'IterativeImputer'). Default is 'KNNImputer'.
    n_neighbors (int): Number of neighbors to use for KNNImputer. Default is 5.
    weights (str): Weight function for KNNImputer. Default is 'uniform'.
    strategy (str): Strategy function for SimpleImputer. Default is 'mean'.
    constant (int/float): Value to fill missing values with when using 'constant'. Default is 0.
    
    Returns:
    pd.DataFrame: DataFrame with imputed values.
    
    Raises:
    ValueError: If an invalid model is specified.
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_columns}")
    
    valid_models = ['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill']
    if model not in valid_models:
        raise ValueError(f"Invalid model specified. Choose from {valid_models}")
    
    feature_columns = df.columns
    print(f'Starting imputation using {model} model')
    
    if model == 'KNNImputer':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df = imputer.fit_transform(df)
    elif model == 'SimpleImputer':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy,fill_value=fill_value)
        df = imputer.fit_transform(df)
    elif model == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42, tol=tol)
        df = imputer.fit_transform(df)
    elif model == 'constant':
        df = df.fillna(constant)
    elif model == 'mean':
        df = df.fillna(df.mean())
    elif model == 'median':
        df = df.fillna(df.median())
    elif model == 'mode':
        df = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    elif model == 'interpolation':
        df = df.interpolate()
    elif model == 'Forward_fill':
        df = df.ffill()
    elif model == 'Backward_fill':
        df = df.bfill()
    
    
    df = pd.DataFrame(df, columns=feature_columns)
    
    print(f'Imputation completed using {model} model')
    return df


def fill_missing_values_column(df: pd.DataFrame, 
                        columns: List[str], 
                        model: Literal['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill'] = 'KNNImputer', 
                        n_neighbors: int = 5, 
                        weights: str = 'uniform', 
                        strategy: str = 'mean', 
                        fill_value = None,
                        estimator = None,
                        max_iter = 10,
                        tol = 0.001,
                        constant: Union[int, float] = 0
                        ) -> pd.DataFrame:
    """
    Impute missing values in the specified columns of the DataFrame using the specified imputation strategy.
    
    Parameters:
    df (pd.DataFrame): DataFrame with missing values.
    columns (list of str): List of column names to impute.
    model (str): Imputation strategy to use ('KNNImputer', 'SimpleImputer', 'constant', 
                'mean', 'median', 'mode', 'interpolation', 'IterativeImputer'). Default is 'mean'.
    n_neighbors (int): Number of neighbors to use for KNNImputer. Default is 5.
    weights (str): Weight function for KNNImputer. Default is 'uniform'.
    strategy (str): Strategy function for SimpleImputer. Default is 'mean'.
    constant (int/float): Value to fill missing values with when using 'constant'. Default is 0.
    
    Returns:
    pd.DataFrame: DataFrame with imputed values.
    
    Raises:
    ValueError: If an invalid model is specified.
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_columns}")
    
    valid_models = ['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill']
    if model not in valid_models:
        raise ValueError(f"Invalid model specified. Choose from {valid_models}")
    
    # Convert single column name to list
    if isinstance(columns, str):
        columns = [columns]
    
    print(f'Starting imputation for columns {columns} using {model} model')
    
    if model == 'KNNImputer':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'SimpleImputer':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy,fill_value=fill_value)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42, tol=tol)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'constant':
        df[columns] = df[columns].fillna(constant)
    elif model == 'mean':
        df[columns] = df[columns].fillna(df[columns].mean())
    elif model == 'median':
        df[columns] = df[columns].fillna(df[columns].median())
    elif model == 'mode':
        df[columns] = df[columns].apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    elif model == 'interpolation':
        df[columns] = df[columns].interpolate()
    elif model == 'Forward_fill':
        df[columns] = df[columns].ffill()
    elif model == 'Backward_fill':
        df[columns] = df[columns].bfill()
    
    
    print(f'Imputation completed for columns {columns} using {model} model')
    return df


def check_outliers(df):
    outliers_df = df.drop(df.select_dtypes(exclude=['int64', 'float64']).columns.tolist(),axis=1)
    # Calculate the first and third quartiles of the data
    q1, q3 = np.percentile(outliers_df, [25, 75])

    # Calculate the IQR
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify the outliers
    outliers = (outliers_df < lower_bound) | (outliers_df > upper_bound)

    # Print the outliers
    print(outliers_df[outliers].count())

    return outliers_df


def box_plot_all_columns(
    data: pd.DataFrame, 
    subplot_row = 3,
    palette: str = 'magma', 
    palette2: str = 'viridis', 
    figsize: tuple = (16, 12), 
    width: float = 0.5, 
    whis: float = 1.5, 
    notch: bool = True, 
    showmeans: bool = True, 
    mean_marker: str = 'o', 
    mean_color: str = 'black', 
    flier_marker: str = 'o', 
    flier_size: int = 8, 
    flier_color: str = 'black', 
    flier_edge_color: str = 'purple', 
    xlabel: str = 'Values', 
    ylabel: str = None, 
    title: str = 'Box Plot', 
    font_scale: float = 1, 
    orient: Optional[Literal['v', 'h', 'x', 'y']] = 'y'
) -> None:
    """
    Create box plots for all numerical columns in the DataFrame using Seaborn with the provided parameters.

    Parameters:
    ----------
    - data: DataFrame
        The dataset for plotting.
    - palette: str
        Color palette for the plot.
    - figsize: tuple
        Size of the figure (width, height).
    - width: float
        Width of the box in the boxplot.
    - whis: float
        Whisker length in terms of IQR.
    - notch: bool
        Whether to draw a notch to indicate the confidence interval.
    - showmeans: bool
        Whether to show the mean value in the plot.
    - mean_marker: str
        Marker style for the mean value.
    - mean_color: str
        Color of the mean marker.
    - flier_marker: str
        Marker style for outliers.
    - flier_size: int
        Size of the outlier markers.
    - flier_color: str
        Color of the outlier markers.
    - flier_edge_color: str
        Edge color of the outlier markers.
    - xlabel: str
        Label for the x-axis.
    - ylabel: str
        Label for the y-axis.
    - title: str
        Title of the plot.
    - font_scale: float
        Scaling factor for the font size of all text elements.
    - orient: {'v', 'h', 'x', 'y'}, optional
        Orientation of the plot (vertical or horizontal).
    
    Returns:
    -------
    - None
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a pandas DataFrame.")

    # Set font scale for all text elements and styling
    sns.set(font_scale=font_scale, style='white')

    # Extract numerical columns
    numerical_cols = data.select_dtypes(include=np.number).columns
    num_cols = len(numerical_cols)

    if num_cols == 0:
        raise ValueError("The DataFrame does not contain any numerical columns.")

    # Set up subplots
    num_rows = int(np.ceil(num_cols / subplot_row))
    fig, axes = plt.subplots(num_rows, subplot_row, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    

    colors = random.sample(sns.color_palette(palette) + sns.color_palette(palette2) + sns.color_palette(palette='gist_gray') + sns.color_palette(palette='tab20b'), num_cols)

    # Loop through each numerical column and plot
    for i, col in enumerate(numerical_cols):
        sns.boxplot(
            data=data, x=col, color=colors[i], width=width, 
            whis=whis, notch=notch, showmeans=showmeans, orient=orient,
            meanprops=dict(marker=mean_marker, markerfacecolor=mean_color, markeredgecolor=mean_color),
            flierprops=dict(marker=flier_marker, markersize=flier_size, 
                            markerfacecolor=flier_color, markeredgecolor=flier_edge_color),
            ax=axes[i]
        )
        axes[i].set_title(f'{title}: {col}', fontsize=14 * font_scale)
        axes[i].set_xlabel(xlabel, fontsize=12 * font_scale)
        # axes[i].set_ylabel(ylabel, fontsize=12 * font_scale)
        axes[i].tick_params(axis='x', labelsize=10 * font_scale)
        axes[i].tick_params(axis='y', labelsize=10 * font_scale)
        axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Suppress Matplotlib INFO messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def check_balance_classification(df: pd.DataFrame, column_plot: Optional[str] = None, palette='magma', edgecolor='black', order: bool = True) -> pd.DataFrame:
    """
    This function takes a DataFrame and a column name, computes the value counts for that column,
    displays the counts as a small DataFrame, and creates a Seaborn count plot with customizations.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    column_plot (str, optional): The column name for which to compute and plot the value counts. If None,
                        the function will use the last column of the DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame showing the value counts for the specified column.
    """
    if column_plot is None:
        column_plot = df.iloc[:, -1].name
    
    if column_plot not in df.columns:
        raise ValueError(f"The column '{column_plot}' does not exist in the DataFrame.")
    
    if order:
        order_list = df[column_plot].value_counts().index
    else:
        order_list = None
    
    # Compute value counts
    value_counts = df[column_plot].value_counts().to_frame()
    value_counts.rename(columns={column_plot: 'value_counts'}, inplace=True)
    value_counts.index.name = 'name'
    
    # Plot the count plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.countplot(data=df, x=column_plot, hue=column_plot, palette=palette, 
                    order=order_list, edgecolor=edgecolor)
    
    # Add percentages on the bars
    total = len(df[column_plot])
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.xlabel(column_plot, fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Count', fontsize=14)  # Set y-axis label with font size
    plt.title('Count Plot', fontsize=16)  # Set title with font size
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.3)  # Add grid lines with custom style
    plt.xticks(fontsize=12)  # Set font size for x-axis ticks
    plt.yticks(fontsize=12)  # Set font size for y-axis ticks
    plt.show()
    
    return value_counts


def get_columns_with_2_unique_values(df):
    object_columns = df.select_dtypes(include=['object']).columns
    columns_with_2_unique_values = [col for col in object_columns if df[col].nunique() == 2]
    
    unique_values_dict = {}
    for col in columns_with_2_unique_values:
        unique_vals = tuple(sorted(df[col].unique()))
        if unique_vals not in unique_values_dict:
            unique_values_dict[unique_vals] = []
        unique_values_dict[unique_vals].append(col)
    
    result_dict = {
        'columns_with_2_unique_name': list(unique_values_dict.values()),
        'columns_with_2_unique_values': list(unique_values_dict.keys())
    }
    
    df_list_of_columns_with_2_unique = pd.DataFrame(result_dict)
    
    for i in range(df_list_of_columns_with_2_unique.shape[0]):
        col_names = df_list_of_columns_with_2_unique["columns_with_2_unique_name"].to_list()[i]
        unique_vals = df_list_of_columns_with_2_unique["columns_with_2_unique_values"].to_list()[i]
        print(f'The list: {col_names} unique values: {unique_vals}')


def analyze_null_columns(df):
    # Calculate null counts for each column
    null_counts = df.isnull().sum()
    
    # Filter out columns with zero null counts
    not_zero_null_counts = null_counts[null_counts > 0]
    
    # Get the data types for the filtered columns
    column_types = df.dtypes[not_zero_null_counts.index]
    
    # Create the final DataFrame
    null_columns = pd.DataFrame({
        'Column': not_zero_null_counts.index,
        'Null Count': not_zero_null_counts.values,
        'Type': column_types.values
    })
    
    # Sort the DataFrame by the 'Type' column
    null_columns.sort_values(by='Type', inplace=True)
    
    # Separate columns by their data types
    null_object_columns = null_columns[null_columns['Type'] == 'object']['Column'].to_list()
    null_numerical_columns = (
        null_columns[null_columns['Type'] == 'float64']['Column'].to_list() + 
        null_columns[null_columns['Type'] == 'int64']['Column'].to_list()
    )
    
    print(f'The columns dtype is object: {null_object_columns}')
    print(f'The columns dtype is numerical: {null_numerical_columns}')
    
    return null_columns


def rate_by_group(
    df, 
    group_col: str, 
    target_col: str, 
    id_col: str, 
    positive_class: int = 1, 
    threshold: int = 5,
    print_output = True
) -> pd.DataFrame:
    """
    Calculate the classification rate by group or range in a given DataFrame.

    Args:
    - df: pandas DataFrame containing the data.
    - group_col: column name by which to group the data.
    - target_col: column name of the target classification.
    - id_col: column name for unique identification of entries.
    - positive_class: value representing the positive class in the target column (default is 1).
    - threshold: the minimum number of unique values in group_col to consider grouping by ranges.

    Returns:
    - A DataFrame with the classification rates by group or range.
    """
    n_unique_values = df[group_col].nunique()
    group_classification_rates = {}

    if n_unique_values <= threshold:
        for group in df[group_col].unique():
            total = df[df[group_col] == group][id_col].nunique()
            classified = df[(df[target_col] == positive_class) & (df[group_col] == group)][id_col].nunique()
            classification_rate = classified / total
            group_classification_rates[group] = classification_rate
            if print_output:
                print(f"The classification rate of {group} in {group_col}: {classification_rate * 100:.2f}%\n")
        
        sorted_groups = sorted(group_classification_rates.keys(), key=lambda x: group_classification_rates[x], reverse=True)
        sorted_classification_rates = {group: group_classification_rates[group] for group in sorted_groups}
        
        return pd.DataFrame(list(sorted_classification_rates.items()), columns=['name', 'values'])
    
    if len(df[group_col].unique()) > threshold and (pd.api.types.is_integer_dtype(df[group_col]) or pd.api.types.is_float_dtype(df[group_col])):
        min_value = df[group_col].min()
        max_value = df[group_col].max()
        bin_size = (max_value - min_value) // threshold
        bins = list(range(min_value, max_value, bin_size)) + [max_value + 1]
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        df['range'] = pd.cut(df[group_col], bins=bins, labels=labels, include_lowest=True, right=False)
        group_col_name = f'{group_col}_range'
        group_col = 'range'
    else:
        group_col_name = group_col
    
    for group in df[group_col].unique():
        total = df[df[group_col] == group][id_col].nunique()
        classified = df[(df[target_col] == positive_class) & (df[group_col] == group)][id_col].nunique()
        classification_rate = classified / total
        group_classification_rates[group] = classification_rate
    
    # Determine sorting method based on data type of group_col
    if pd.api.types.is_numeric_dtype(df[group_col]):
        sorted_groups = sorted(group_classification_rates.keys(), key=lambda x: float(x.split('-')[0]) if '-' in x else float(x))
    else:
        sorted_groups = sorted(group_classification_rates.keys(), key=lambda x: group_classification_rates[x], reverse=True)
    
    for group in sorted_groups:
        classification_rate = group_classification_rates[group]
        if print_output:
            print(f"The classification rate of {group} in {group_col_name}: {classification_rate * 100:.2f}%\n")
    
    group_of_range = {group: group_classification_rates[group] for group in sorted_groups}
    
    return pd.DataFrame(list(group_of_range.items()), columns=['name', 'values'])



def over_under_sampling_classification(
    x: pd.DataFrame, 
    y: pd.Series, 
    over_sampling: Literal['SMOTE', 'SVMSMOTE', 'BorderlineSMOTE-1', 'BorderlineSMOTE-2', 'ADASYN', 'SMOTEN', 'SMOTENC', 'random_over_sampler'] = 'SVMSMOTE', 
    under_sampling: Literal['TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN', 'CondensedNearestNeighbour', 'ClusterCentroids', 'NearMiss', 'random_under_sampler'] = 'TomekLinks', 
    over_sampling_strategy= "auto", 
    under_sampling_strategy= "auto", 
    k_neighbors: int = 5, 
    random_state: int = 42, 
    categorical_features: Optional[list] = None, 
    over: bool = True, 
    under: bool = True,
    make_df: bool = True,
    n_jobs = -1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print(f'Starting over-sampling and/or under-sampling process.')
    print(f'Initial class distribution: {Counter(y)}')
    
    valid_over_sampling_strategies = [
        'random_over_sampler', 'SMOTE', 'SVMSMOTE', 'BorderlineSMOTE-1', 
        'BorderlineSMOTE-2', 'ADASYN', 'SMOTEN', 'SMOTENC'
    ]
    valid_under_sampling_strategies = [
        'random_under_sampler', 'TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 
        'AllKNN', 'CondensedNearestNeighbour', 'ClusterCentroids', 'NearMiss'
    ]
    
    if over and over_sampling not in valid_over_sampling_strategies:
        raise ValueError(f"Invalid over_sampling strategy '{over_sampling}' specified. "
                        f"Valid options are: {', '.join(valid_over_sampling_strategies)}")
    
    if under and under_sampling not in valid_under_sampling_strategies:
        raise ValueError(f"Invalid under_sampling strategy '{under_sampling}' specified. "
                        f"Valid options are: {', '.join(valid_under_sampling_strategies)}")
    
    print(f'\nuse {over_sampling} model for oversampling')
    
    # Over-sampling
    if over:
        if over_sampling == 'SMOTE':
            print(f'Applying SMOTE with strategy {over_sampling_strategy}')
            smote = SMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
            x, y = smote.fit_resample(x, y)
        elif over_sampling == 'SVMSMOTE':
            print(f'Applying SVMSMOTE with strategy {over_sampling_strategy}')
            svmsmote = SVMSMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = svmsmote.fit_resample(x, y)
        elif over_sampling == 'BorderlineSMOTE-1':
            print(f'Applying BorderlineSMOTE(kind="borderline-1") with strategy {over_sampling_strategy}')
            bl1smote = BorderlineSMOTE(kind='borderline-1', sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = bl1smote.fit_resample(x, y)
        elif over_sampling == 'BorderlineSMOTE-2':
            print(f'Applying BorderlineSMOTE(kind="borderline-2") with strategy {over_sampling_strategy}')
            bl2smote = BorderlineSMOTE(kind='borderline-2', sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = bl2smote.fit_resample(x, y)
        elif over_sampling == 'ADASYN':
            print(f'Applying ADASYN with strategy {over_sampling_strategy}')
            adasyn = ADASYN(sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = adasyn.fit_resample(x, y)
        elif over_sampling == 'SMOTEN':
            print(f'Applying SMOTEN with strategy {over_sampling_strategy}')
            smoten = SMOTEN(sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = smoten.fit_resample(x, y)
        elif over_sampling == 'SMOTENC':
            if categorical_features is None:
                raise ValueError("categorical_features must be provided for SMOTENC")
            print(f'Applying SMOTENC with strategy {over_sampling_strategy}')
            smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy=over_sampling_strategy, random_state=random_state, n_jobs=n_jobs)
            x, y = smotenc.fit_resample(x, y)
        elif over_sampling == 'random_over_sampler':
            print(f'Applying RandomOverSampler with strategy {over_sampling_strategy}')
            ros = RandomOverSampler(sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = ros.fit_resample(x, y)
            
    print(f'after oversampling class distribution: {Counter(y)}')
    print(f'\nuse {under_sampling} model for undersampling')
    # Under-sampling
    if under:
        if under_sampling == 'TomekLinks':
            print(f'Applying TomekLinks under-sampling.')
            tom = TomekLinks(n_jobs=n_jobs)
            x, y = tom.fit_resample(x, y)
        elif under_sampling == 'EditedNearestNeighbours':
            print(f'Applying EditedNearestNeighbours with strategy {under_sampling_strategy}')
            enn = EditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', n_jobs=n_jobs)
            x, y = enn.fit_resample(x, y)
        elif under_sampling == 'RepeatedEditedNearestNeighbours':
            print(f'Applying RepeatedEditedNearestNeighbours with strategy {under_sampling_strategy}')
            renn = RepeatedEditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, max_iter=100, kind_sel='all', n_jobs=n_jobs)
            x, y = renn.fit_resample(x, y)
        elif under_sampling == 'AllKNN':
            print(f'Applying AllKNN with strategy {under_sampling_strategy}')
            allknn = AllKNN(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', allow_minority=True, n_jobs=n_jobs)
            x, y = allknn.fit_resample(x, y)
        elif under_sampling == 'CondensedNearestNeighbour':
            print(f'Applying CondensedNearestNeighbour with strategy {under_sampling_strategy}')
            cnn = CondensedNearestNeighbour(sampling_strategy=under_sampling_strategy, n_neighbors=1, random_state=random_state, n_jobs=n_jobs)
            x, y = cnn.fit_resample(x, y)
        elif under_sampling == 'ClusterCentroids':
            print(f'Applying ClusterCentroids with strategy {under_sampling_strategy}')
            cc = ClusterCentroids(sampling_strategy=under_sampling_strategy, random_state=random_state, voting='soft')
            x, y = cc.fit_resample(x, y)
        elif under_sampling == 'NearMiss':
            print(f'Applying NearMiss(version=1) with strategy {under_sampling_strategy}')
            nm = NearMiss(sampling_strategy=under_sampling_strategy, version=1, n_neighbors=3, n_jobs=n_jobs)
            x, y = nm.fit_resample(x, y)
        elif under_sampling == 'random_under_sampler':
            print(f'Applying RandomUnderSampler with strategy {under_sampling_strategy}')
            rus = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=random_state)
            x, y = rus.fit_resample(x, y)
    
    print(f'after undersampling class distribution: {Counter(y)}')
    
    print(f'\nFinal class distribution: {Counter(y)}')
    print(f'Over-sampling and/or under-sampling process completed.')
    if make_df:
        x_resampled = pd.DataFrame(x, columns=x.columns)
        y_resampled = pd.Series(y, name=y.name)
        combined_df = pd.concat([x_resampled, y_resampled], axis=1)
        return combined_df, x_resampled, y_resampled
    else:
        return x, y



def custom_smote_regression(x: pd.DataFrame, y: pd.Series, sampling_strategy: float, k_neighbors: int, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_state)
    n_samples = int(sampling_strategy * len(y))
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(x)
    
    synthetic_x = []
    synthetic_y = []
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(y))
        sample_x, sample_y = x.iloc[idx], y.iloc[idx]
        _, indices = neighbors.kneighbors([sample_x])
        neighbor_idx = np.random.choice(indices[0])
        neighbor_x, neighbor_y = x.iloc[neighbor_idx], y.iloc[neighbor_idx]
        
        diff_x = neighbor_x - sample_x
        new_x = sample_x + np.random.rand() * diff_x
        new_y = sample_y + np.random.rand() * (neighbor_y - sample_y)
        
        synthetic_x.append(new_x)
        synthetic_y.append(new_y)
    
    synthetic_x = pd.DataFrame(synthetic_x, columns=x.columns)
    synthetic_y = pd.Series(synthetic_y, name=y.name)
    
    return pd.concat([x, synthetic_x]), pd.concat([y, synthetic_y])

def over_under_sampling_regression(x: pd.DataFrame, 
                                   y: pd.Series, 
                                   over_sampling: Optional[str] = 'custom_smote', 
                                   under_sampling: Optional[str] = 'random_under_sampler', 
                                   over_sampling_strategy: Union[float, dict] = 0.5, 
                                   under_sampling_strategy: Union[float, dict] = 0.5, 
                                   k_neighbors: int = 5, 
                                   random_state: int = 42, 
                                   over: bool = True, 
                                   under: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Perform over-sampling and/or under-sampling on the given dataset for regression.

    Parameters:
    x (pd.DataFrame): The input features.
    y (pd.Series): The target variable.
    over_sampling (str, optional): The over-sampling strategy to use ('custom_smote'). Default is 'custom_smote'.
    under_sampling (str, optional): The under-sampling strategy to use ('random_under_sampler'). Default is 'random_under_sampler'.
    over_sampling_strategy (float or dict): The strategy to use for over-sampling. Default is 0.5.
    under_sampling_strategy (float or dict): The strategy to use for under-sampling. Default is 0.5.
    k_neighbors (int): Number of nearest neighbors for custom SMOTE. Default is 5.
    random_state (int): Random state for reproducibility. Default is 42.
    over (bool): Whether to apply over-sampling. Default is True.
    under (bool): Whether to apply under-sampling. Default is True.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: The DataFrame with combined features and target, resampled features, and resampled target.
    """
    logging.info('Starting over-sampling and/or under-sampling process.')
    
    # Over-sampling
    if over:
        if over_sampling == 'custom_smote':
            logging.info('Applying custom SMOTE with strategy %s', over_sampling_strategy)
            x, y = custom_smote_regression(x, y, sampling_strategy=over_sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
        else:
            raise ValueError(f"Invalid over_sampling strategy '{over_sampling}' specified.")
    
    # Under-sampling
    if under:
        if under_sampling == 'random_under_sampler':
            logging.info('Applying RandomUnderSampler with strategy %s', under_sampling_strategy)
            rus = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=random_state)
            x, y = rus.fit_resample(x, y)
        else:
            raise ValueError(f"Invalid under_sampling strategy '{under_sampling}' specified.")
    
    # Combine the resampled features and target into a single DataFrame
    df = pd.concat([x, y], axis=1)
    
    logging.info('Over-sampling and/or under-sampling process completed.')

    return df, x, y


# def over_under_sampling_regression(
#     x: pd.DataFrame, 
#     y: pd.Series, 
#     over_sampling: Optional[str] = None, 
#     under_sampling: Optional[str] = None, 
#     over_sampling_strategy: Union[float, dict] = 0.5, 
#     under_sampling_strategy: Union[float, dict] = 0.5, 
#     k_neighbors: int = 5, 
#     random_state: int = 42, 
#     over: bool = True, 
#     under: bool = True
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
#     """
#     Perform over-sampling and/or under-sampling on the given dataset for regression.

#     Parameters:
#     x (pd.DataFrame): The input features.
#     y (pd.Series): The target variable.
#     over_sampling (str, optional): The over-sampling strategy to use ('random_over_sampler', 'smote', 'adasyn', or None for SMOTE). Default is None.
#     under_sampling (str, optional): The under-sampling strategy to use ('random_under_sampler', 'tomek_links', 'enn', 'renn', 'allknn', 
#                                     'cnn', 'cc', 'nm'). Default is None.
#     over_sampling_strategy (float or dict): The strategy to use for over-sampling. Default is 0.5.
#     under_sampling_strategy (float or dict): The strategy to use for under-sampling. Default is 0.5.
#     k_neighbors (int): Number of nearest neighbors for SMOTE. Default is 5.
#     random_state (int): Random state for reproducibility. Default is 42.
#     over (bool): Whether to apply over-sampling. Default is True.
#     under (bool): Whether to apply under-sampling. Default is True.

#     Returns:
#     Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: The DataFrame with combined features and target, resampled features, and resampled target.
#     """
#     print(f'Starting over-sampling and/or under-sampling process.')
    
#     valid_over_sampling_strategies = [
#         'random_over_sampler', 'smote', 'adasyn'
#     ]
#     valid_under_sampling_strategies = [
#         'random_under_sampler', 'tomek_links', 'enn', 'renn', 'allknn', 
#         'cnn', 'cc', 'nm'
#     ]
    
#     if over and over_sampling not in valid_over_sampling_strategies:
#         raise ValueError(f"Invalid over_sampling strategy '{over_sampling}' specified. "
#                          f"Valid options are: {', '.join(valid_over_sampling_strategies)}")
    
#     if under and under_sampling not in valid_under_sampling_strategies:
#         raise ValueError(f"Invalid under_sampling strategy '{under_sampling}' specified. "
#                          f"Valid options are: {', '.join(valid_under_sampling_strategies)}")
    
#     # Over-sampling
#     if over:
#         if over_sampling == 'smote':
#             print(f'Applying SMOTE with strategy {over_sampling_strategy}')
#             smote = SMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
#             x, y = smote.fit_resample(x, y)
#         elif over_sampling == 'adasyn':
#             print(f'Applying ADASYN with strategy {over_sampling_strategy}')
#             adasyn = ADASYN(sampling_strategy=over_sampling_strategy, random_state=random_state)
#             x, y = adasyn.fit_resample(x, y)
#         elif over_sampling == 'random_over_sampler':
#             print(f'Applying RandomOverSampler with strategy {over_sampling_strategy}')
#             ros = RandomOverSampler(sampling_strategy=over_sampling_strategy, random_state=random_state)
#             x, y = ros.fit_resample(x, y)
    
#     # Under-sampling
#     if under:
#         if under_sampling == 'tomek_links':
#             print(f'Applying TomekLinks under-sampling.')
#             tom = TomekLinks(n_jobs=-1)
#             x, y = tom.fit_resample(x, y)
#         elif under_sampling == 'enn':
#             print(f'Applying EditedNearestNeighbours with strategy {under_sampling_strategy}')
#             enn = EditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', n_jobs=-1)
#             x, y = enn.fit_resample(x, y)
#         elif under_sampling == 'renn':
#             print(f'Applying RepeatedEditedNearestNeighbours with strategy {under_sampling_strategy}')
#             renn = RepeatedEditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, max_iter=100, kind_sel='all', n_jobs=-1)
#             x, y = renn.fit_resample(x, y)
#         elif under_sampling == 'allknn':
#             print(f'Applying AllKNN with strategy {under_sampling_strategy}')
#             allknn = AllKNN(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', allow_minority=True, n_jobs=-1)
#             x, y = allknn.fit_resample(x, y)
#         elif under_sampling == 'cnn':
#             print(f'Applying CondensedNearestNeighbour with strategy {under_sampling_strategy}')
#             cnn = CondensedNearestNeighbour(sampling_strategy=under_sampling_strategy, n_neighbors=1, random_state=random_state, n_jobs=-1)
#             x, y = cnn.fit_resample(x, y)
#         elif under_sampling == 'cc':
#             print(f'Applying ClusterCentroids with strategy {under_sampling_strategy}')
#             cc = ClusterCentroids(sampling_strategy=under_sampling_strategy, random_state=random_state, voting='soft')
#             x, y = cc.fit_resample(x, y)
#         elif under_sampling == 'nm':
#             print(f'Applying NearMiss(version=1) with strategy {under_sampling_strategy}')
#             nm = NearMiss(sampling_strategy=under_sampling_strategy, version=1, n_neighbors=3, n_jobs=-1)
#             x, y = nm.fit_resample(x, y)
#         elif under_sampling == 'random_under_sampler':
#             print(f'Applying RandomUnderSampler with strategy {under_sampling_strategy}')
#             rus = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=random_state)
#             x, y = rus.fit_resample(x, y)
    
#     # Combine the resampled features and target into a single DataFrame
#     df = pd.concat([x, y], axis=1)
    
#     print(f'Over-sampling and/or under-sampling process completed.')
    
#     return df, x, y




def check_Balance_Regression(df: pd.DataFrame, column_plot: Optional[str] = None, div_number: int = 4) -> pd.DataFrame:
    """
    Check the balance of a regression target variable by binning the data and plotting the distribution.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_plot (str, optional): The column name for the regression target variable. If None,
                        the function will use the last column of the DataFrame.
    div_number (int): The number of bins to divide the target variable into. Default is 4.
    
    Returns:
    pd.DataFrame: A DataFrame showing the binned data and bin labels.
    
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_plot is None:
        column_plot = df.iloc[:, -1].name
    
    if column_plot not in df.columns:
        raise ValueError(f"The column '{column_plot}' does not exist in the DataFrame.")
    
    logging.info('Binning the data for column: %s into %d bins', column_plot, div_number)
    
    # Bin the data
    binned_data, bin_edges = pd.cut(df[column_plot], bins=div_number, retbins=True, labels=False, ordered=True)
    
    # Create readable bin labels
    bin_labels = [f'{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]
    temp_df = pd.DataFrame({
        'binned': binned_data,
        'bin_labels': binned_data.map(lambda x: bin_labels[int(x)] if pd.notna(x) else 'NaN')
    })
    
    logging.info('Creating count plot for binned data of column: %s', column_plot)
    
    # Plot the data using seaborn countplot
    plt.figure(figsize=(10, 6))  # Set the figure size
    palette = sns.color_palette("magma", len(bin_labels))
    ax= sns.countplot(data=temp_df, x='bin_labels', hue='bin_labels', palette=palette, order=bin_labels, edgecolor='black', legend=False)
    plt.title(f'Count plot of binned regression output for {column_plot}')
    plt.xlabel('Bins', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Number of Data Points', fontsize=14)  # Set y-axis label with font size
    plt.title('Count Plot', fontsize=16)  # Set title with font size
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.3)  # Add grid lines with custom style
    plt.xticks(fontsize=12, rotation=45) # Set font size for x-axis ticks
    plt.yticks(fontsize=12)  # Set font size for y-axis ticks
    # Create custom legend
    handles = [Patch(color=palette[(len(bin_labels)-1)-i], label=bin_labels[i]) for i in range(len(bin_labels))]
    ax.legend(handles=handles, title='Bins', loc='upper right')
    plt.show()
    
    # return temp_df

def plot_groupby(df: pd.DataFrame, groupby_cols: List[str], plot: bool = True) -> pd.DataFrame:
    """
    Groups the dataframe by specified columns, computes the mean for numeric columns, and optionally plots the results.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    groupby_cols (List[str]): The columns to group by.
    plot (bool): Whether to plot the results. Default is True.

    Returns:
    pd.DataFrame: The grouped dataframe with mean values.
    """

    def compute_additional_metrics(df_grouped: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        df_grouped['MCC'] = df_grouped.apply(lambda row: matthews_corrcoef(df[col1], df[col2]), axis=1)
        df_grouped['Balanced Accuracy'] = df_grouped.apply(lambda row: balanced_accuracy_score(df[col1], df[col2]), axis=1)
        return df_grouped

    try:
        # Grouping and calculating mean
        df_group = df[groupby_cols]
        df_grouped = df_group.groupby(groupby_cols[:-1], as_index=False).mean()

        # Compute additional metrics if applicable
        if len(groupby_cols) == 2:
            df_grouped = compute_additional_metrics(df_grouped, groupby_cols[0], groupby_cols[-1])

        # Plotting the grouped data
        if plot:
            fig, ax = plt.subplots()
            ax.plot(df_grouped[groupby_cols[0]], df_grouped[groupby_cols[-1]])
            ax.set_xlabel(groupby_cols[0])
            ax.set_ylabel('Mean of ' + groupby_cols[-1])
            ax.set_title('Groupby Plot')
            plt.show()

        return df_grouped

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def plot_pivot(df_group_one, index_col, columns_col):
    grouped_pivot = df_group_one.pivot(index=index_col, columns=columns_col)
    print(grouped_pivot)
    
    # Plotting the pivoted data
    fig, ax = plt.subplots()
    im = ax.pcolor(grouped_pivot, cmap='RdBu')
    
    row_labels = grouped_pivot.columns.levels[1]
    col_labels = grouped_pivot.index
    
    ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)
    
    fig.colorbar(im)
    plt.show()

    return grouped_pivot

def plot_pivot_2(df: pd.DataFrame, index_col: str, columns_col: str, values_col: Optional[str] = None, plot: bool = True) -> pd.DataFrame:
    """
    Pivots the dataframe by specified index and columns, computes the mean for numeric columns, and optionally plots the results.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    index_col (str): The column to use as index.
    columns_col (str): The column to use as columns.
    values_col (Optional[str]): The column to use as values. If None, all remaining columns will be used. Default is None.
    plot (bool): Whether to plot the results. Default is True.

    Returns:
    pd.DataFrame: The pivoted dataframe.
    """

    def compute_additional_metrics(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        df['MCC'] = df.apply(lambda row: matthews_corrcoef(df[col1], df[col2]), axis=1)
        df['Balanced Accuracy'] = df.apply(lambda row: balanced_accuracy_score(df[col1], df[col2]), axis=1)
        return df

    try:
        if values_col:
            df_pivot = df.pivot(index=index_col, columns=columns_col, values=values_col)
        else:
            df_pivot = df.pivot(index=index_col, columns=columns_col)
        
        # Compute additional metrics if applicable
        if values_col and len(df[values_col].unique()) == 2:
            df_pivot = compute_additional_metrics(df_pivot, index_col, columns_col)

        if plot:
            fig, ax = plt.subplots()
            im = ax.pcolor(df_pivot, cmap='RdBu')
            
            row_labels = df_pivot.columns
            col_labels = df_pivot.index
            
            ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
            
            ax.set_xticklabels(row_labels, minor=False)
            ax.set_yticklabels(col_labels, minor=False)
            
            fig.colorbar(im)
            plt.show()

        return df_pivot

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def plot_groupby_and_pivot(df: pd.DataFrame, groupby_cols: List[str], index_col: str, columns_col: str, plot_group: bool = True, plot_pivot: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Groupby and pivot a DataFrame, then optionally plot the results.

    Parameters:
    - df: The DataFrame to process.
    - groupby_cols: List of columns to group by. The last column is assumed to be the one to aggregate.
    - index_col: Column to use as the new index in the pivoted DataFrame.
    - columns_col: Column to use to create the new columns in the pivoted DataFrame.
    - plot_group: Whether to plot the groupby results (default is True).
    - plot_pivot: Whether to plot the pivot results (default is True).

    Returns:
    - A dictionary containing the grouped and pivoted DataFrames.

    Example usage:
    results = plot_groupby_and_pivot(df, ['col1', 'col2'], 'col1', 'col2', plot_group=True, plot_pivot=True)
    grouped_df = results['grouped']
    pivoted_df = results['pivoted']
    """
    
    def plot_grouped_data(df_grouped: pd.DataFrame, x_col: str, y_col: str) -> None:
        """
        Plot the grouped data.

        Parameters:
        - df_grouped: Grouped DataFrame to plot.
        - x_col: Column for x-axis.
        - y_col: Column for y-axis.
        """
        try:
            fig, ax = plt.subplots()
            ax.plot(df_grouped[x_col], df_grouped[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel('Mean of ' + y_col)
            ax.set_title('Groupby Plot')
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting grouped data: {e}")

    def plot_pivot_data(pivoted_df: pd.DataFrame) -> None:
        """
        Plot the pivoted data.

        Parameters:
        - pivoted_df: Pivoted DataFrame to plot.
        """
        try:
            fig, ax = plt.subplots()
            im = ax.pcolor(pivoted_df, cmap='RdBu')

            row_labels = pivoted_df.columns.levels[1]
            col_labels = pivoted_df.index

            ax.set_xticks(np.arange(pivoted_df.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(pivoted_df.shape[0]) + 0.5, minor=False)

            ax.set_xticklabels(row_labels, minor=False)
            ax.set_yticklabels(col_labels, minor=False)

            fig.colorbar(im)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting pivoted data: {e}")

    # Group the data
    try:
        df_group = df[groupby_cols]
        df_group_one = df_group.groupby(groupby_cols[:-1], as_index=False).mean()
        print("Grouped DataFrame:\n", df_group_one)
    except Exception as e:
        print(f"An error occurred during grouping: {e}")
        return {}

    # Plot the grouped data if requested
    if plot_group:
        plot_grouped_data(df_group_one, groupby_cols[0], groupby_cols[-1])

    # Pivot the grouped data
    try:
        grouped_pivot = df_group_one.pivot(index=index_col, columns=columns_col)
        print("Pivoted DataFrame:\n", grouped_pivot)
    except Exception as e:
        print(f"An error occurred during pivoting: {e}")
        return {'grouped': df_group_one}

    # Plot the pivoted data if requested
    if plot_pivot:
        plot_pivot_data(grouped_pivot)

    return {'grouped': df_group_one, 'pivoted': grouped_pivot}

def calculate_correlation(df: pd.DataFrame, outcome_column: Optional[str] = None, num_results: Optional[int] = 5) -> pd.DataFrame:
    """
    Calculates and prints the Pearson correlation coefficient and p-value for each numeric column in the DataFrame
    against the specified outcome column, ordered by the Pearson correlation coefficient.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing data.
    outcome_column (str): The name of the outcome column to calculate the correlation against.
    num_results (int, optional): The number of top results to display. If None, display all results.

    Returns:
    pd.DataFrame: A DataFrame containing the Pearson correlation coefficients and p-values for each numeric column.
    """
    if outcome_column is None:
        outcome_column = df.iloc[:, -1].name
    
    if outcome_column not in df.columns:
        raise ValueError(f"The column '{outcome_column}' does not exist in the DataFrame.")
    
    # Select numeric columns from the DataFrame
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    
    # Ensure the outcome column is included in the numeric columns
    if outcome_column not in numeric_columns:
        raise ValueError(f"The outcome column '{outcome_column}' must be numeric and present in the DataFrame.")
    
    print('Calculating Pearson correlation coefficients for numeric columns against the outcome column: {outcome_column}')
    
    # Store the results
    results = []

    # Loop through each numeric column and calculate Pearson correlation
    for param in numeric_columns:
        if param != outcome_column:
            pearson_coef, p_value = stats.pearsonr(df[param].dropna(), df[outcome_column].dropna())
            results.append((param, pearson_coef, p_value))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Variable', 'Pearson Coefficient', 'P-Value'])

    # Order the results by Pearson correlation coefficient
    results_df = results_df.reindex(results_df['Pearson Coefficient'].abs().sort_values(ascending=False).index)
    
    # Limit the number of results if num_results is specified
    if num_results is not None:
        results_df = results_df.head(num_results)
    
    print(f'Top {num_results if num_results is not None else len(results_df)} results:\n{results_df}')
    
    # Print the results
    for index, row in results_df.iterrows():
        print(f"\n{row['Variable']}")
        print(f"The Pearson Correlation Coefficient for {row['Variable']} is {row['Pearson Coefficient']:.4f} with a P-value of P = {row['P-Value']:.4g}")
    
    return results_df

def Heatmap_Correlation(df: pd.DataFrame, mask: float = 0, cmap: str = "YlGnBu", save_path: Optional[str] = None, annot_size = 10, figsize=(20, 14)) -> pd.DataFrame:
    """
    Generates a heatmap to visualize the correlation matrix of numeric columns in the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    mask (float, optional): Threshold to mask correlations below this value. Default is 0.
    save_path (str, optional): Path to save the heatmap image. If None, the heatmap is not saved. Default is None.

    Returns:
    pd.DataFrame: Correlation matrix of the numeric columns in the DataFrame.
    """
    # Select numeric columns from the DataFrame
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_columns.empty:
        raise ValueError("The DataFrame does not contain any numeric columns.")
    
    # Compute the correlation matrix
    correlations = numeric_columns.corr()
    
    # Create a figure and axis for the heatmap
    plt.figure(figsize=figsize)
    
    # Create a heatmap with customization
    heatmap = sns.heatmap(
        data=correlations,
        annot=True,                     # Annotate cells with the data value
        fmt=".2f",                      # Format the annotations to 2 decimal places
        cmap=cmap,                      # Colormap
        cbar=True,                      # Show color bar
        cbar_kws={'label': 'Scale'},    # Color bar customization
        linewidths=0.5,                 # Line width between cells
        linecolor='gray',               # Line color between cells
        square=True,                    # Force square cells
        mask=correlations < mask,       # Mask correlations below the threshold
        annot_kws={"size": annot_size}, # Annotation font size 
        xticklabels=True,               # Show x-axis labels
        yticklabels=True,               # Show y-axis labels
        robust=True                     # Robust colormap limits
    )
    
    # Customize the plot
    plt.title('Heatmap of Correlation Matrix', fontsize=18, weight='bold', pad=20)
    plt.xlabel('Features', fontsize=14, weight='bold')
    plt.ylabel('Features', fontsize=14, weight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.3)
    
    # Save the heatmap if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info('Heatmap saved to %s', save_path)
    
    # Display the heatmap
    plt.show()
    
    # Return the correlation matrix
    return correlations

def create_custom_scatter_plot(df: pd.DataFrame, 
                               x_col: str, 
                               y_col: str, 
                               hue_col: Optional[str] = None, 
                               title: str = 'Scatter Plot', 
                               xlabel: Optional[str] = None, 
                               ylabel: Optional[str] = None, 
                               size_range: Tuple[int, int] = (20, 500), 
                               alpha: float = 0.7, 
                               palette: str = 'viridis') -> None:
    """
    Create a customized scatter plot using Seaborn.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    x_col (str): The column name for the x-axis values.
    y_col (str): The column name for the y-axis values.
    hue_col (str, optional): The column name for the hue (color coding). Default is None.
    title (str): The title of the plot. Default is 'Scatter Plot'.
    xlabel (str, optional): The label for the x-axis. If None, use the column name. Default is None.
    ylabel (str, optional): The label for the y-axis. If None, use the column name. Default is None.
    size_range (tuple): The range of sizes for the scatter plot points. Default is (20, 500).
    alpha (float): The transparency level of the points. Default is 0.7.
    palette (str): The color palette to use for the plot. Default is 'viridis'.

    Returns:
    None
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"The specified columns '{x_col}' and/or '{y_col}' do not exist in the DataFrame.")
    
    if hue_col and hue_col not in df.columns:
        raise ValueError(f"The specified hue column '{hue_col}' does not exist in the DataFrame.")
    
    if hue_col is None:
        hue_col = y_col
    
    logging.info('Creating scatter plot for %s vs %s', x_col, y_col)
    
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col,
        palette=palette, 
        sizes=size_range, 
        alpha=alpha, 
        edgecolor='w', 
        linewidth=1.5
    )
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel(xlabel if xlabel else x_col, fontsize=14)
    plt.ylabel(ylabel if ylabel else y_col, fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.3)
    
    if hue_col:
        plt.legend(title=hue_col, fontsize=10, title_fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    logging.info('Scatter plot created successfully.')

def plot_histograms(df: pd.DataFrame, column: Optional[Union[str, None]] = None, save_plots: bool = False, palette: str = 'magma',
                            bins: Optional[Union[int, list]] = 30, edgecolor: str = 'black', alpha: float = 0.9, kde: bool = True,
                            multiple: Literal['layer', 'dodge', 'stack', 'fill'] = 'layer', show = True, single_histogram_figsize = (8, 6), all_histograms_figsize = (14, 12)) -> None:
    """
    Plots histograms for numerical columns in the dataframe using Seaborn.
    
    Parameters:
    ----------
    df (pd.DataFrame): The input dataframe.
    column (Union[str, None], optional): The specific column to plot. If None, plot all numerical columns.
    save_plots (bool, optional): Whether to save the plots as images. Default is False.
    palette (str, optional): The color palette to use for the plots. Default is 'magma'.
    bins (Union[int, list], optional): Number of bins or bin edges for the histogram. Default is None.
    edgecolor (str, optional): Color of the edge of the bins. Default is 'black'.
    alpha (float, optional): Transparency level of the bins. Default is 0.9.
    kde (bool, optional): Whether to plot a KDE. Default is True.
    multiple (Literal['layer', 'dodge', 'stack', 'fill'], optional): How to plot multiple elements. Default is 'layer'.
    
    Examples:
    >>> plot_histograms_seaborn(penguins, bins=30)
    >>> plot_histograms_seaborn(penguins,"body_mass_g", bins=30)
    
    Returns:
    -------
    None
    """
    
    def plot_single_histogram(col_data: pd.Series, col_name: str, color: str) -> None:
        plt.figure(figsize=single_histogram_figsize)
        sns.histplot(col_data, kde=kde, color=color, bins=bins, edgecolor=edgecolor, alpha=alpha, multiple=multiple)
        plt.title(f"Histogram for {col_name}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"histogram_{col_name}.png")
        plt.show()
    
    def plot_all_histograms(numerical_cols: pd.DataFrame) -> None:
        num_columns = len(numerical_cols.columns)
        ncols = int(np.ceil(np.sqrt(num_columns)))
        nrows = int(np.ceil(num_columns / ncols))
        colors = sns.color_palette(palette, num_columns)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=all_histograms_figsize)
        for index, col in enumerate(numerical_cols.columns):
            ax = axes.flatten()[index]
            sns.histplot(df[col], kde=kde, color=colors[index], bins=bins, edgecolor=edgecolor, alpha=alpha, multiple=multiple, ax=ax)
            ax.set_title(col)

        # Remove empty subplots
        for i in range(num_columns, nrows * ncols):
            fig.delaxes(axes.flatten()[i])

        plt.suptitle("Histograms of Numerical Columns", size=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            plt.savefig("all_histograms.png")
        if show:
            plt.show()
    
    try:
        numerical_columns = df.select_dtypes(include=['int64', 'float64'])
        
        if column is None or column.lower() == 'all':
            plot_all_histograms(numerical_columns)
        else:
            if column in numerical_columns.columns:
                color = sns.color_palette(palette, 1)[0]
                plot_single_histogram(df[column], column, color)
            else:
                print(f"Column '{column}' is not a numerical column in the dataframe.")
    except Exception as e:
        print(f"An error occurred: {e}")


def encode_column(
    df: pd.DataFrame, 
    columns: Union[str, List[str]], 
    method: Literal[
        'get_dummies', 'label', 'ordinal', 'binary', 'target', 'dict_vectorizer', 
        'feature_hasher', 'label_binarizer', 'multi_label_binarizer', 'frequency'
    ] = 'get_dummies', 
    ordinal_categories: Optional[List[str]] = None, 
    target: Optional[str] = None, 
    n_features: Optional[int] = None,
    binary_1: str = None,
    binary_0: str = None,
    binary_default: bool = True
) -> pd.DataFrame:
    """
    Encodes one or more columns in the dataframe using the specified method.

    Parameters:
    ----------
    df (pd.DataFrame): The dataframe containing the column(s) to be encoded.
    columns (Union[str, List[str]]): The name of the column or a list of column names to be encoded.
    method (Literal['get_dummies', 'label', 'ordinal', 'binary', 'target', 
                    'dict_vectorizer', 'feature_hasher', 'label_binarizer', 
                    'multi_label_binarizer', 'frequency']): The encoding method to use. 
                    Options are 'get_dummies' (one_hot), 'label', 'ordinal', 
                    'binary', 'target', 'dict_vectorizer', 'feature_hasher', 
                    'label_binarizer', 'multi_label_binarizer', 'frequency'. Default is 'get_dummies'.
    ordinal_categories (Optional[List[str]]): Categories for ordinal encoding if method is 'ordinal'. Default is None.
    target (Optional[str]): Target column for target encoding. Default is None.
    n_features (Optional[int]): Number of features for feature hasher. Default is None.
    
    Examples:
    --------
    One-hot encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='get_dummies')
    Ordinal encoding with specified categories
    >>> df_encoded = encode_column(df, 'column_name', method='ordinal', ordinal_categories=['low', 'medium', 'high'])
    Binary encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='binary')
    Target encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='target', target='target_column')
    Feature hashing with a specified number of features
    >>> df_encoded = encode_column(df, 'column_name', method='feature_hasher', n_features=10)
    Frequency encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='frequency')

    Returns:
    -------
    pd.DataFrame: The dataframe with the encoded column(s).

    """
    
    def binary_encode(df: pd.DataFrame, column: str, binary_1: str = None, binary_0: str = None) -> pd.DataFrame:
        unique_vals = df[column].dropna().unique()
        if len(unique_vals) != 2:
            raise ValueError("Column must have exactly two unique non-NaN values.")
        if binary_1 is not None and binary_0 is not None:
            df[column] = df[column].apply(lambda x: 1 if x == binary_1 else (0 if x == binary_0 else np.nan))
        elif binary_1 not in unique_vals or binary_0 not in unique_vals or binary_1 is None or binary_0 is None:
            df[column] = df[column].apply(lambda x: 1 if x == unique_vals[0] else (0 if x == unique_vals[1] else np.nan))
        return df

    def label_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        le = LabelEncoder()
        non_nan_mask = df[column].notna()
        le.fit(df.loc[non_nan_mask, column])
        df.loc[non_nan_mask, column] = le.transform(df.loc[non_nan_mask, column])
        return df

    def ordinal_encode(df: pd.DataFrame, column: str, categories: List[str]) -> pd.DataFrame:
        oe = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=np.nan)
        nan_mask = df[column].isna()
        df[column] = oe.fit_transform(df[[column]])
        df.loc[nan_mask, column] = np.nan
        return df

    def get_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
        nan_mask = df[column].isna()
        dummies = pd.get_dummies(df[column], prefix=column, prefix_sep='_', drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        df.loc[nan_mask, dummies.columns] = pd.NA
        df = df.drop(column, axis=1)
        return df

    def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        te = TargetEncoder(cols=[column])
        df[column] = te.fit_transform(df[column], df[target])
        return df

    def dict_vectorize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        dv = DictVectorizer(sparse=False)
        dict_data = df[column].apply(lambda x: {column: x})
        transformed = dv.fit_transform(dict_data)
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=dv.get_feature_names_out())], axis=1)
        return df

    def feature_hash(df: pd.DataFrame, column: str, n_features: int) -> pd.DataFrame:
        fh = FeatureHasher(n_features=n_features, input_type='string')
        transformed = fh.transform(df[column].astype(str)).toarray()
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=[f'{column}_hashed_{i}' for i in range(n_features)])], axis=1)
        return df

    def label_binarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        lb = LabelBinarizer()
        transformed = lb.fit_transform(df[column])
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=lb.classes_)], axis=1)
        return df

    def multi_label_binarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()
        transformed = mlb.fit_transform(df[column])
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=mlb.classes_)], axis=1)
        return df
    
    def frequency_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        freq = df[column].value_counts() / len(df)
        df[column] = df[column].map(freq)
        return df

    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe")
        
        if binary_default:
            unique_vals = df[column].value_counts()
            if len(unique_vals) == 2:
                df = binary_encode(df, column)
                continue

        if method == 'binary':
            df = binary_encode(df, column)
        elif method == 'label':
            df = label_encode(df, column)
        elif method == 'ordinal':
            if ordinal_categories is None:
                raise ValueError("ordinal_categories must be provided for ordinal encoding")
            df = ordinal_encode(df, column, ordinal_categories)
        elif method == 'get_dummies':
            df = get_dummies(df, column)
        elif method == 'target':
            if target is None:
                raise ValueError("Target column must be provided for target encoding")
            df = target_encode(df, column, target)
        elif method == 'dict_vectorizer':
            df = dict_vectorize(df, column)
        elif method == 'feature_hasher':
            if n_features is None:
                raise ValueError("Number of features must be provided for feature hasher")
            df = feature_hash(df, column, n_features)
        elif method == 'label_binarizer':
            df = label_binarize(df, column)
        elif method == 'multi_label_binarizer':
            df = multi_label_binarize(df, column)
        elif method == 'frequency':
            df = frequency_encode(df, column)
        else:
            raise ValueError(f"Encoding method '{method}' is not supported")
    
    return df


def get_x_y(df,target:Optional[str] = None):
    if target == None:
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]
    else:
        x=df.drop(target,axis=1)
        y=df[target]
    return x,y


def get_x_y_TT(df: pd.DataFrame, target: Optional[str] = None, test_size: float = 0.2, stratify: bool = False, shuffle: bool = True, random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the dataframe into training and test sets.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target (Optional[str]): The target column. If None, the last column is used as the target.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    stratify (bool): Whether to stratify based on the target column. Default is False.
    shuffle (bool): Whether or not to shuffle the data before splitting. Default is True.
    random_state (Optional[int]): Controls the shuffling applied to the data before splitting. Pass an int for reproducible output. Default is 42.
    
    Returns:
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Returns x_train, y_train, x_test, y_test.
    """
    
    def split_data_TT(df: pd.DataFrame, target: Optional[str], test_size: float, stratify: bool, shuffle: bool, random_state: Optional[int]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Splits the data into training and test sets."""
        x, y = extract_features_and_target(df, target)
        stratify_col = None
        if stratify:
            if y.nunique() < 20:  # This is a simple heuristic; you might want to adjust it
                stratify_col = y
            else:
                print("Stratification is not applicable for regression data. Continuing without stratification.")
                stratify = False
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=stratify_col, shuffle=shuffle, random_state=random_state)
        return x_train, y_train, x_test, y_test
    
    def extract_features_and_target(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extracts features and target from the dataframe."""
        if target is None:
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        else:
            x = df.drop(target, axis=1)
            y = df[target]
        return x, y

    try:
        return split_data_TT(df, target, test_size, stratify, shuffle, random_state)
    except Exception as e:
        raise ValueError(f"An error occurred while splitting the data: {e}")


def get_x_y_TVT(df: pd.DataFrame, target: Optional[str] = None, test_size: float = 0.2, valid_size: float = 0.12, stratify: bool = False, shuffle: bool = True, random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the dataframe into training, validation, and test sets.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target (Optional[str]): The target column. If None, the last column is used as the target.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    train_size (float): The proportion of the dataset to include in the train split. Default is 0.8.
    valid_size (float): The proportion of the training dataset to include in the validation split. Default is 0.12.
    stratify (bool): Whether to stratify based on the target column. Default is False.
    shuffle (bool): Whether or not to shuffle the data before splitting. Default is True.
    random_state (Optional[int]): Controls the shuffling applied to the data before splitting. Pass an int for reproducible output. Default is 42.
    
    Returns:
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Returns x_train, y_train, x_valid, y_valid, x_test, y_test.
    """
    
    def split_data_TVT(df: pd.DataFrame, target: Optional[str], test_size: float, valid_size: float, stratify: bool, shuffle: bool, random_state: Optional[int]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Splits the data into training, validation, and test sets."""
        x, y = extract_features_and_target(df, target)
        stratify_col = None
        if stratify:
            if y.nunique() < 20:  # This is a simple heuristic; you might want to adjust it
                stratify_col = y
            else:
                print("Stratification is not applicable for regression data. Continuing without stratification.")
                stratify = False
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=stratify_col, shuffle=shuffle, random_state=random_state)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, shuffle=shuffle, random_state=random_state)
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    
    def extract_features_and_target(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extracts features and target from the dataframe."""
        if target is None:
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        else:
            x = df.drop(target, axis=1)
            y = df[target]
        return x, y

    try:
        return split_data_TVT(df, target, test_size, valid_size, stratify, shuffle, random_state)
    except Exception as e:
        raise ValueError(f"An error occurred while splitting the data: {e}")


def get_x_y_TT_shape(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Prints the shapes of training, validation, and test datasets.

    Parameters:
    x_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    x_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test labels.

    Returns:
    None
    """
    
    def print_shape(data: pd.DataFrame, label: str) -> None:
        """Prints the shape of a given dataset."""
        print(f'{label} shape = {data.shape}')
    
    try:
        print_shape(x_train, 'x_train')
        print_shape(x_test, 'x_test')
        print_shape(y_train, 'y_train')
        print_shape(y_test, 'y_test')
    except Exception as e:
        raise ValueError(f"An error occurred while printing shapes: {e}")


def get_x_y_TVT_shape(x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Prints the shapes of training, validation, and test datasets.

    Parameters:
    x_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    x_valid (pd.DataFrame): The validation features.
    y_valid (pd.Series): The validation labels.
    x_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test labels.

    Returns:
    None
    """
    
    def print_shape(data: pd.DataFrame, label: str) -> None:
        """Prints the shape of a given dataset."""
        print(f'{label} shape = {data.shape}')
    
    try:
        print_shape(x_train, 'x_train')
        print_shape(x_valid, 'x_valid')
        print_shape(x_test, 'x_test')
        print_shape(y_train, 'y_train')
        print_shape(y_valid, 'y_valid')
        print_shape(y_test, 'y_test')
    except Exception as e:
        raise ValueError(f"An error occurred while printing shapes: {e}")


def validate_test_data_categorical_columns(x_train, x_test, x_valid):
    x_train_categorical_columns = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    x_test_categorical_columns = x_test.select_dtypes(include=['object', 'category']).columns.tolist()
    x_valid_categorical_columns = x_valid.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if set(x_train_categorical_columns) != set(x_test_categorical_columns) or set(x_train_categorical_columns) != set(x_valid_categorical_columns):
        print('Train, test, and validation dataframes have different categorical columns')
        print('Train categorical columns:', x_train_categorical_columns)
        print('Test categorical columns:', x_test_categorical_columns)
        print('Validation categorical columns:', x_valid_categorical_columns)
        return
    
    all_columns_consistent = True
    
    for cat_col in x_train_categorical_columns:
        if cat_col not in x_test_categorical_columns or cat_col not in x_valid_categorical_columns:
            print(f'Column {cat_col} is missing in test or validation DataFrame')
            all_columns_consistent = False
            continue
        
        train_col = set(x for x in x_train[cat_col].unique().tolist() if not pd.isna(x))
        test_col = set(x for x in x_test[cat_col].unique().tolist() if not pd.isna(x))
        valid_col = set(x for x in x_valid[cat_col].unique().tolist() if not pd.isna(x))
        
        if train_col != test_col or train_col != valid_col:
            print(f'{cat_col} column has different unique values in train, test, and validation data:')
            print(f'Unique values in train data: {train_col}')
            print(f'Unique values in test data: {test_col}')
            print(f'Unique values in validation data: {valid_col}')
            all_columns_consistent = False
    
    if all_columns_consistent:
        print('All categorical columns have consistent unique values in train, test, and validation data.')
    else:
        print('Some categorical columns have inconsistencies. Please review the above differences.')


def scale_data(x_train: Union[np.ndarray, pd.DataFrame], 
               x_test: Union[np.ndarray, pd.DataFrame], 
               x_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
               scaler_type: Literal['standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'l2', 'log'] = 'standard') -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Scales the input data using the specified scaler type.
    
    Parameters:
    x_train (Union[np.ndarray, pd.DataFrame]): Training data.
    x_test (Union[np.ndarray, pd.DataFrame]): Test data.
    x_valid (Optional[Union[np.ndarray, pd.DataFrame]]): Validation data (optional).
    scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'l2', 'log'). Default is 'standard'.
    
    Returns:
    Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: Returns the scaled (x_train, x_test) if x_valid is not provided,
    otherwise returns (x_train, x_valid, x_test).
    """
    
    def get_scaler(scaler_type: str, n_samples: int):
        """Returns the scaler object based on the scaler type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'maxabs':
            return MaxAbsScaler()
        elif scaler_type == 'quantile':
            return QuantileTransformer(output_distribution='uniform', n_quantiles=min(n_samples, 1000))
        elif scaler_type == 'power':
            return PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'l2':
            return Normalizer(norm='l2')
        elif scaler_type == 'log':
            return None  # Log transformation handled separately
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

    def log_transform(*arrays):
        """Applies log transformation to the given arrays."""
        return tuple(np.log1p(array) for array in arrays)
    
    try:
        n_samples = x_train.shape[0]
        scaler = get_scaler(scaler_type, n_samples)
        
        if scaler_type == 'log':
            if x_valid is None:
                return log_transform(x_train, x_test)
            else:
                return log_transform(x_train, x_valid, x_test)
        
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        if x_valid is not None:
            x_valid_scaled = scaler.transform(x_valid)
            return x_train_scaled, x_test_scaled, x_valid_scaled
        else:
            return x_train_scaled, x_test_scaled
    
    except Exception as e:
        raise ValueError(f"An error occurred while scaling the data: {e}")

def get_cross_validator(cv_type: Literal['KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'] = 'KFold', 
                        cv=5, shuffle=True, random_state=42, LeavePOut_p=2, RepeatedKFold_n_repeats=10):
    if cv_type == 'KFold':
        type_cross_valid = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'StratifiedKFold':
        type_cross_valid = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'LeaveOneOut':
        type_cross_valid = LeaveOneOut()
    elif cv_type == 'LeavePOut':
        type_cross_valid = LeavePOut(p=LeavePOut_p)
    elif cv_type == 'RepeatedKFold':
        type_cross_valid = RepeatedKFold(n_splits=cv, n_repeats=RepeatedKFold_n_repeats, random_state=random_state)
    elif cv_type == 'TimeSeriesSplit':
        type_cross_valid = TimeSeriesSplit(n_splits=cv)
    else:
        raise ValueError("Invalid cv_type. Choose from 'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'.")
    return type_cross_valid


def get_hidden_layer_activation(
    activation_type: Literal['sigmoid', 'tanh', 'relu', 'LeakyReLU', 'PReLU', 'ELU', 'swish'] = 'relu',
    LeakyReLU_alpha: float = 0.1,
    ELU_alpha: float = 1.0
):
    """
    Returns a specified Keras activation function.

    Parameters:
    - activation_type (Literal): The type of activation function to use. Default is 'relu'.
    - LeakyReLU_alpha (float): Slope of the negative section for LeakyReLU. Default is 0.1.
    - ELU_alpha (float): Slope of the negative section for ELU. Default is 1.0.

    Returns:
    - activation: A Keras activation function or layer.
    """

    if activation_type == 'sigmoid':
        activation = 'sigmoid'
    elif activation_type == 'tanh':
        activation = 'tanh'
    elif activation_type == 'relu':
        activation = 'relu'
    elif activation_type == 'LeakyReLU':
        activation = LeakyReLU(alpha=LeakyReLU_alpha)
    elif activation_type == 'PReLU':
        activation = PReLU()
    elif activation_type == 'ELU':
        activation = ELU(alpha=ELU_alpha)
    elif activation_type == 'swish':
        activation = Activation('swish')
    else:
        raise ValueError("Invalid activation_type. Choose from 'sigmoid', 'tanh', 'relu', 'LeakyReLU', 'PReLU', 'ELU', 'swish'.")
    
    return activation


def get_output_layer_activation(
    activation_type: Literal['linear', 'tanh', 'relu', 'sigmoid', 'softmax'] = 'sigmoid',
):
    """
    Returns a specified Keras activation function.

    Parameters:
    - activation_type (Literal): The type of activation function to use. Default is 'sigmoid'.

    Returns:
    - activation: A Keras activation function or layer.
    """

    if activation_type == 'linear':
        activation = 'linear'
    elif activation_type == 'tanh':
        activation = 'tanh'
    elif activation_type == 'relu':
        activation = 'relu'
    elif activation_type == 'sigmoid':
        activation = 'sigmoid'
    elif activation_type == 'softmax':
        activation = 'softmax'
    else:
        raise ValueError("Invalid activation_type. Choose from 'linear', 'tanh', 'relu', 'sigmoid', 'softmax'.")
    
    return activation


def get_optimizer(
    op_type: Literal['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam', 'Adadelta', 'Adamax', 'Nadam'] = 'Adam',
    learning_rate: float = 0.001,
    SGD_learning_rate: float = 0.01,
    momentum: float = 0.9,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    decay: float = 0.0,
    amsgrad: bool = False,
    rho: float = 0.9,
    initial_accumulator_value: float = 0.1
):
    """
    Returns a specified Keras optimizer.

    Parameters:
    - op_type (Literal): The type of optimizer to use. Default is 'Adam'.
    - learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    - SGD_learning_rate (float): Learning rate specifically for SGD-based optimizers. Default is 0.01.
    - momentum (float): Momentum factor for optimizers that use momentum. Default is 0.9.
    - beta_1 (float): The exponential decay rate for the 1st moment estimates in Adam-based optimizers. Default is 0.9.
    - beta_2 (float): The exponential decay rate for the 2nd moment estimates in Adam-based optimizers. Default is 0.999.
    - decay (float): Learning rate decay over each update for some optimizers. Default is 0.0.
    - amsgrad (bool): Whether to apply the AMSGrad variant of the Adam optimizer. Default is False.
    - rho (float): Discounting factor for the history/coming gradient in RMSprop and Adadelta. Default is 0.9.
    - initial_accumulator_value (float): Starting value for the accumulators in Adagrad. Default is 0.1.

    Returns:
    - optimizer: A Keras optimizer instance.
    """

    if op_type == 'SGD':
        optimizer = SGD(learning_rate=SGD_learning_rate)
    elif op_type == 'Momentum':
        optimizer = SGD(learning_rate=SGD_learning_rate, momentum=momentum)
    elif op_type == 'Nesterov':
        optimizer = SGD(learning_rate=SGD_learning_rate, momentum=momentum, nesterov=True)
    elif op_type == 'Adagrad':
        optimizer = Adagrad(learning_rate=learning_rate, initial_accumulator_value=initial_accumulator_value)
    elif op_type == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate, rho=rho, decay=decay)
    elif op_type == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)
    elif op_type == 'Adadelta':
        optimizer = Adadelta(learning_rate=learning_rate, rho=rho, decay=decay)
    elif op_type == 'Adamax':
        optimizer = Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
    elif op_type == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
    else:
        raise ValueError("Invalid optimizer_type. Choose from 'SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam', 'Adadelta', 'Adamax', 'Nadam'.")
    return optimizer


def get_loss(
    loss_type: Literal['mean_squared_error', 'mean_absolute_error', 'Huber', 
                        'binary_crossentropy', 'hinge', 'categorical_crossentropy', 
                        'sparse_categorical_crossentropy'] = 'mean_squared_error',
    from_logits: bool = False,
    delta: float = 1.0,
    reduction: Literal['none', 'sum', 'sum_over_batch_size'] = 'sum_over_batch_size',
    label_smoothing: float = 0.0,
    axis: int = -1
):
    """
    Returns a specified Keras loss function.

    Parameters:
    - loss_type (Literal): The type of loss function to use. Default is 'mean_squared_error'.
    - from_logits (bool): Whether to interpret the output as logits (for cross-entropy losses). Default is False.
    - delta (float): The threshold at which to change between quadratic and linear loss (for Huber loss). Default is 1.0.
    - reduction (Literal): Type of reduction to apply to the loss. Options are 'none', 'sum', 'sum_over_batch_size'. Default is 'sum_over_batch_size'.
    - label_smoothing (float): Float in [0, 1]. When > 0, label values are smoothed, meaning the confidence on label values are relaxed. Default is 0.0.
    - axis (int): The axis along which to compute crossentropy (for categorical crossentropy losses). Default is -1.

    Returns:
    - loss: A Keras loss function instance.
    """

    if loss_type == 'mean_squared_error':
        loss = MeanSquaredError(reduction=reduction)
    elif loss_type == 'mean_absolute_error':
        loss = MeanAbsoluteError(reduction=reduction)
    elif loss_type == 'Huber':
        loss = Huber(delta=delta, reduction=reduction)
    elif loss_type == 'binary_crossentropy':
        loss = BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction)
    elif loss_type == 'hinge':
        loss = Hinge(reduction=reduction)
    elif loss_type == 'categorical_crossentropy':
        loss = CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction, axis=axis)
    elif loss_type == 'sparse_categorical_crossentropy':
        loss = SparseCategoricalCrossentropy(from_logits=from_logits, reduction=reduction)
    else:
        raise ValueError("Invalid loss_type. Choose from 'mean_squared_error', 'mean_absolute_error', 'Huber', 'binary_crossentropy', 'hinge', 'categorical_crossentropy', 'sparse_categorical_crossentropy'.")
    
    return loss


def plot_feature_importance(
    model: Any, 
    x_train: Any, 
    feature_names: Optional[List[str]] = None, 
    title: Optional[str] = None,
    palette: str = 'viridis', 
    top_n: Literal[None,'first','last'] = None, # 'first' for top 10, 'last' for bottom 10, None for all
    orientation: str = 'vertical',
    figsize: Optional[tuple] = (10, 6),
    bar_width: float = 0.8
) -> None:
    """
    Plot feature importances for tree-based models or coefficients for linear models.

    Parameters:
        model (Any): The trained model.
        x_train (Any): Training data features.
        feature_names (Optional[List[str]]): List of feature names. If not provided, will use x_train's column names or default to indices.
        title (Optional[str]): Title of the plot. Automatically set based on `top_n` if not provided.
        palette (str): Color palette for the plot.
        top_n (Optional[str]): 'first' for top 10 features, 'last' for bottom 10 features, None for all features.
        orientation (str): Orientation of the plot ('vertical' or 'horizontal').
        figsize (Optional[tuple]): Size of the figure.
        bar_width (float): Width of the bars.
    """
    try:
        # Determine feature importances or coefficients
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:  # Multiclass classification case
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = model.coef_
        else:
            print("Model does not have feature importances or coefficients.")
            return

        # Use provided feature names or fall back to x_train's columns or indices
        if feature_names is None:
            feature_names = x_train.columns if hasattr(x_train, 'columns') else np.arange(x_train.shape[1])

        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]
        importances = importances[sorted_indices]
        feature_names = np.array(feature_names)[sorted_indices]

        # Select top_n or bottom_n features
        if top_n == 'first':
            importances = importances[:10]
            feature_names = feature_names[:10]
            plot_title = 'Top 10 Feature Importances'
        elif top_n == 'last':
            importances = importances[-10:][::-1] # Reverse order for last 10
            feature_names = feature_names[-10:][::-1]
            plot_title = 'The Worst 10 Feature Importances'
        else:
            plot_title = 'Feature Importances'

        # Use provided title if available
        plot_title = title if title else plot_title

        # Create the plot
        sns.set(style="whitegrid")
        plt.figure(figsize=figsize)
        if orientation == 'vertical':
            sns.barplot(x=importances, y=feature_names, palette=palette, ci=None, width=bar_width)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Many numbers on x-axis
        else:
            sns.barplot(y=importances, x=feature_names, palette=palette, ci=None, width=bar_width)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Many numbers on y-axis

        plt.title(plot_title)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting: {e}")


def plots_evaluate_models(
    data, 
    labels=None, 
    categories=None,
    have_overfitting=None,
    colors=None,
    title=None, 
    xlabel=None, 
    ylabel=None, 
    palette='magma', 
    width=0.75, 
    edgecolor='black',
    linewidth=1.5,
    hatches: list = None,
    hatch: bool = True,
    order: bool = False,
    figsize=(10, 8),
    data_labels=True,
    annote_num=1

):
    if xlabel is None:
        xlabel = 'Category'
    if ylabel is None:
        ylabel = 'Values'
    if title is None:
        title = 'Multiple Bar Plots'
    
    if not isinstance(data, pd.DataFrame):
        if isinstance(data[0], (list, np.ndarray)):
            df = pd.DataFrame(data).T
        else:
            df = pd.DataFrame(data, columns=['Values'])
    else:
        df = data.copy()
    
    if labels:
        df.columns = labels
    
    if categories:
        df['Category'] = categories
    else:
        df['Category'] = [f'Category {i+1}' for i in range(len(df))]
    
    df_melted = df.melt(id_vars='Category', var_name='Dataset', value_name='Value')
    
    plt.figure(figsize=figsize)
    
    if hatch and not hatches:
        hatches_list = random.sample(['X', 'oo', 'O|', '/', '+', '++', '--', '-\\', 'xx', '*-', '\\\\', '|*', '\\', 'OO', 'o', '**', 'o-', '*', '//', '||', '+o', '..', '/o', 'O.', '\\|', 'x*', '|', '-', None], len(df['Category'].unique()))
    else:
        hatches_list = hatches
    
    if order:
        ordered_x = df_melted.groupby('Category')['Value'].sum().sort_values(ascending=False).index
        ax = sns.barplot(data=df_melted, x='Category', y='Value', hue='Dataset', palette=colors if colors else palette, errorbar=None, order=ordered_x, width=width)
    else:
        ax = sns.barplot(data=df_melted, x='Category', y='Value', hue='Dataset', palette=colors if colors else palette, errorbar=None, width=width)
    
    for bar in ax.patches:
        bar.set_edgecolor(edgecolor)
        bar.set_linewidth(linewidth)
    
    if hatch and hatches_list:
        for i, bar_group in enumerate(ax.patches):
            hatch_pattern = hatches_list[i % len(hatches_list)]
            bar_group.set_hatch(hatch_pattern)
    
    if data_labels:
        num_categories = len(categories)
        for i, p in enumerate(ax.patches):
            if have_overfitting:
                color_index = i // num_categories
                if color_index < len(have_overfitting):
                    if have_overfitting[color_index] < 0:
                        color = 'red'
                    elif have_overfitting[color_index] > 0:
                        color = 'green'
                    else:
                        color = 'black'
                else:
                    color = 'black'
            else:
                color = 'black'
            
            ax.annotate(format(p.get_height(), f'.{annote_num}f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='center right')
    
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    
    plt.show()
