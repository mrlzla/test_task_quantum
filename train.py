import os
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def train(df:pd.DataFrame, cv:int = 5) -> LinearRegression:
    """Train input data only for '6', '7' columns and their polynomial
    features. The exploratory data analysis shows, that we have pretty strong
    relation between '6' and '7' columns and target.

    Args:
        df (pd.DataFrame): train data
        cv (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        LinearRegression: trained regression model
    """
    X = df.drop('target', axis=1)
    y = df['target']

    features = ['6', '7']
    X_part = X[features]

    poly = PolynomialFeatures(2)
    poly_f = poly.fit_transform(X_part)
    poly_cols = poly.get_feature_names_out()
    X_poly = pd.DataFrame(poly_f, columns=poly_cols)
    reg = LinearRegression()
    res = cross_val_score(reg, X_poly, y, cv=cv,
                          scoring='neg_root_mean_squared_error')
    print(f"RMSE metrics for cv={cv}: ", list(-1*res))
    res_model = reg.fit(X_poly, y)
    return res_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV File Processing")
    parser.add_argument("--input_csv",
                        type=str,
                        help="Input CSV file")
    parser.add_argument("--output_folder",
                        type=str,
                        help="A folder where the output model will be stored",
                        default="./output_data")
    parser.add_argument("--output_model_name",
                        type=str,
                        help="Output trained model name",
                        default="regression_model.sav")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    res_model = train(df)
    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(args.output_folder, args.output_model_name)
    pickle.dump(res_model, open(output_path, 'wb'))
    print(f"The result model has been saved to {output_path}")
