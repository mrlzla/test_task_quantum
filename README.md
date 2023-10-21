# Test Task Quantum
Here you can find the solutions for 3 test tasks

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Scikit-learn (for the Random Forest model)
- torchvision (for MNIST dataset)

### Installation

1. Clone this repository.

```shell
git clone https://github.com/mrlzla/test_task_quantum.git
cd test_task_quantum
```

2. Create an environment (optional).

```shell
virtualenv -p python3.10 venv
source venv/bin/activate
```

3. Install the required Python packages.

```shell
pip install -r requirements.txt
```

## Task 1: Counting Islands
This code is a Python implementation for counting the number of islands in a given 2D matrix using the Breadth-First Search (BFS) algorithm. Islands are represented as '1's, oceans as '0's, and visited islands as '2's. The script accepts user input for the matrix dimensions and the matrix itself and then calculates the number of islands present in the map.

### Usage

1. Run the script.
```shell
python counting_islands.py
```

2. Enter the dimensions of the matrix (M and N) as integers separated by a space.

3. Enter the matrix rows. Each row should consist of [0, 1] separated by spaces.

4. The script will output the number of islands in the provided matrix.

### Example

Suppose you input the following matrix:

```
4 4
1 1 0 0
0 1 0 1
1 0 0 1
1 0 0 0
```

The script will output:

```
3
```

This means there are three islands in the provided matrix.

## Task 2: Regression on the tabular data
We have a dataset (train.csv) that contains 53 anonymized features and a target
column. The task is to build a model that predicts a target based on the proposed
features.

Here are three files provided:
- **regression_exploring.ipynb**: Jupyter notebook with exploratory data analysis.
- **train.py**: Python script for training LinearRegression.
- **predict.py**: Python script for model inference on test data.

### Usage

#### Data exploration

Here's how you can see the data exploration:

```shell
jupyter-notebook regression_exploring.ipynb
```

#### Train

In order to train `LinearRegression`:

```
python train.py --input_csv train.csv
```

- `--input_csv`: The path to the input CSV file.
- `--output_folder`: The folder where the trained model will be stored (default is "./output_data").
- `--output_model_name`: The name of the output trained model file (default is "regression_model.sav").

#### Predict
In order to predict model:

```
python predict.py --input_csv hidden_test.csv
```

- `--input_csv`: The path to the input CSV file.
- `--output_folder`: The folder where the output CSV file will be stored (default is "./output_data").
- `--output_csv_name`: The name of the output CSV file (default is "hidden_test_result.csv").

There is no need to use pretrained model, because the relation between input columns and target is pretty clear.

## Task 3: Digit Classifier

This is a Python program for digit classification using various model types. It includes an abstract interface for digit classification models and a `DigitClassifier` class that dynamically selects and uses the specified model for digit classification.

### Usage

Here's how you can run the digit classifier:

```shell
python digit_classifier.py --model_type cnn
```

- `--model_type`: Specify the model type for digit classification. Deault is 'cnn'.
- `--output_folder`: Provide a folder path where the MNIST dataset will be stored. Default is '. output_data'. 

Replace `<model_type>` with the desired model type ('cnn', 'rf', or 'rand') and `<output_folder>` with the path where the MNIST dataset will be stored.

#### Models

The program supports three types of models for digit classification:
- **CNN (Convolutional Neural Network)**: Uses a simple PyTorch CNN model for inference.
- **Random Forest (RF)**: Utilizes the scikit-learn RandomForestClassifier for classification.
- **Random Model (RAND)**: Generates random predictions for demonstration purposes.

Each model type corresponds to a different class that implements the `DigitClassificationInterface`.