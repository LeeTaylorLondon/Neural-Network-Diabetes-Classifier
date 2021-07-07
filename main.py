import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing


def load_diabetes():
    return pd.read_csv('diabetes.csv')

def info_diabetes_csv(df):
    print(df.head())
    df.hist()
    plt.show()

def replace_0(df):
    """ Replace missing info fields with np.nan """
    df['Glucose'] = df['Glucose'].replace(0, np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
    df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
    df['Insulin'] = df['Insulin'].replace(0, np.nan)
    df['BMI'] = df['BMI'].replace(0, np.nan)
    return df

def replace_mean(df):
    """ Replace missing info fields with mean values """
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
    return df

def standardize(df):
    df_scale = preprocessing.scale(df)
    # df is no longer of type DataFrame - convert it back
    df_scale = pd.DataFrame(df_scale, columns=df.columns)
    # Scale back outcome column
    df_scale['Outcome'] = df['Outcome']
    df = df_scale
    print(df.describe().loc[['mean', 'std', 'max'],].round(2).abs())
    return df


if __name__ == '__main__':
    df = load_diabetes()
    df = replace_0(df)
    df = replace_mean(df)
    df = standardize(df)