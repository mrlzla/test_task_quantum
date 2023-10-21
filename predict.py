import os
import argparse
import pandas as pd


def inference(df: pd.DataFrame) -> pd.DataFrame:
    """Predicts the target column based on exploratory data analysis.
    The exploratory data analysis shows, that we have pretty strong relation
    between '6' and '7' columns and target.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: CSV File with predicted target column
    """
    res = df['7'] + df['6']**2
    res_df = pd.DataFrame(res, columns=['target'])
    return res_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV File Processing")
    parser.add_argument("--input_csv",
                        type=str,
                        help="Input CSV file")
    parser.add_argument("--output_folder",
                        type=str,
                        help="A folder where the output CSV will be stored",
                        default="./output_data")
    parser.add_argument("--output_csv_name",
                        type=str,
                        help="Output CSV file name",
                        default="hidden_test_result.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(args.output_folder, args.output_csv_name)
    res = inference(df)
    res.to_csv(output_path)
    print(f"The result CSV has been saved to {output_path}")
