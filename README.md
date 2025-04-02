# Mushroom Classification using Deep Learning

This project focuses on classifying mushrooms as edible or poisonous using deep learning techniques, specifically Convolutional Neural Networks (CNNs). Accurate classification is crucial for foraging and food safety.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Extracted Insights](#extracted-insights)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The project utilizes the Mushroom Classification dataset from Kaggle, which includes descriptions of 8,143 mushroom samples with 23 features, each labeled as either edible or poisonous. citeturn0search6

**Key Steps in the Analysis:**

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Handle categorical variables through encoding.
   - Split the data into training and testing sets.
   - Scale features to ensure uniformity.

2. **Model Building:**
   - Construct a CNN using TensorFlow and Keras.
   - Define the architecture with convolutional, pooling, and fully connected layers.
   - Compile the model with appropriate loss functions and optimizers.

3. **Model Training:**
   - Train the model on the training data.
   - Monitor performance metrics during training.

4. **Model Evaluation:**
   - Evaluate the model's performance on the test set.
   - Use metrics such as accuracy, precision, recall, and F1-score.

5. **Prediction:**
   - Utilize the trained model to predict mushroom classifications.
   - Interpret the results to identify at-risk mushrooms.

## Project Structure

The repository contains the following files:

- `mushroom-classification-using-deep-learning.ipynb`: A Jupyter Notebook that performs data preprocessing, model building, training, and evaluation.
- `mushrooms.csv`: The dataset file containing mushroom samples and their features.

## Setup Instructions

To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   Use the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/dattatejaofficial/Mushroom-Classification.git
   ```

2. **Navigate to the Project Directory:**
   Move into the project directory:

   ```bash
   cd Mushroom-Classification
   ```

3. **Create a Virtual Environment (optional but recommended):**
   Set up a virtual environment to manage project dependencies:

   ```bash
   python3 -m venv env
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using pip:

   ```bash
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   ```

## Usage

To run the analysis:

1. **Ensure the Virtual Environment is Activated:**
   Make sure your virtual environment is active (refer to the setup instructions above).

2. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open `mushroom-classification-using-deep-learning.ipynb` in the Jupyter interface and execute the cells sequentially to perform the analysis.

## Extracted Insights

The application of deep learning to the Mushroom Classification dataset provides the following insights:

- **Model Performance:** The CNN model achieves high accuracy, effectively distinguishing between edible and poisonous mushrooms.

- **Feature Importance:** The model identifies key features influencing mushroom classification, aiding in understanding the characteristics of different mushroom species.

## Dependencies

The project requires the following Python packages:

- `tensorflow`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

These dependencies are essential for data manipulation, model building, and visualization tasks.
