# Deep Learning Based Recommender System

## Description
This project is an online movie recommendation system powered by Restricted Boltzmann Machines (RBM). It offers users personalized movie recommendations by analyzing their past viewing behavior and predicting the top 10 movies they might enjoy.

## Prerequisites
- Python 3.x
- TensorFlow library
- Numpy
- Pandas
- MovieLens dataset (or any suitable dataset)

## Setup
1. Clone this repository to your local machine.

2. Install the required dependencies:

## Usage
1. **Download and preprocess the dataset:**
- You will need a movie dataset, such as the MovieLens dataset. Ensure it includes user ratings and movie information.

2. **Prepare the dataset:**
- Load and preprocess the dataset, cleaning and encoding the data as needed. Sample code for this is provided in the `data_preparation.py` file.

3. **Train the RBM model:**
- Run the `train.py` script to train the RBM model on your preprocessed dataset. Adjust hyperparameters as needed.

4. **Make recommendations:**
- Use the `recommend.py` script to generate personalized movie recommendations for users. The script will predict the top 10 movies for each user based on their viewing history and the trained RBM model.

### Example Usage
```bash
python data_preparation.py
python train.py
python recommend.py
