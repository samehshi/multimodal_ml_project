# Classifying Student Engagement with Multimodal Data

This repository contains the code, documentation, and trained models for a project that classifies student engagement levels (high, medium, low) using multimodal data: facial expressions, speech emotions, heart rate, and overall confidence. The project utilizes machine learning techniques, including Random Forest, Support Vector Machines (SVMs), Neural Networks (MLPs) and an ensemble model, to predict engagement levels.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)

## Project Overview

This project explores the potential of multimodal data analysis for real-time, objective student engagement assessment in classrooms. By combining information from different modalities, we aim to develop a robust model that can effectively identify varying levels of student engagement.

The core of the project is an evaluation of different classification models to determine which approach performs best, using an F1-score metric. The models are tested with and without hyperparameter tuning to understand the performance impact. The models include Random Forest, SVMs, MLPs and an ensemble model.

## Repository Structure

```
├── Code.ipynb                        # Jupyter notebook for the project
├── Paper_Presentation.pptx           # PowerPoint slides for project presentation
├── Paper_Final.docx                # Final report/paper in word format
├── Ten-students_combined_data.xlsx  # The dataset used in the project (ensure this file is excluded in git)
├── README.md                        # This file
├── best_rf_model.pkl               # Trained Random Forest model
├── best_nn_model.pkl                # Trained Neural Network model
├── best_svm_model.pkl                # Trained Support Vector Machine model
```

## Data

The dataset used for this project is a multimodal dataset composed of:

*   Facial expression scores (e.g. `face_neutral`, `face_happy`).
*   Speech emotion scores (e.g. `speech_neutral`, `speech_angry`).
*   Heart Rate measurements.
*   Overall confidence scores.
*   Engagement levels (high, medium, low)

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/samehshi/multimodal_ml_project.git
    cd multimodal_ml_project
    ```

2. **Set up the Environment:** You may need to install specific packages for your system, depending on your setup and project needs. Please ensure you have the libraries required in the project to be able to execute the code.

## Usage

To reproduce the results and explore the project:

1.  **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Code.ipynb
   ```
2.  **Execute Cells:** Step through the notebook by running the code cells sequentially. This will perform EDA, preprocessing, model training, and model evaluation.
3.  **Experiment:** Feel free to modify the code, try different hyperparameters, or explore alternative modeling approaches.

## Model Training and Evaluation

The Jupyter notebook (`Code.ipynb`) contains the following steps:

*   **Exploratory Data Analysis (EDA):** The notebook performs a thorough EDA, visualizing feature distributions, examining relationships between variables, and conducting correlation analyses. The results of this analysis are visualized and documented within the notebook.
*   **Data Preprocessing:** The notebook handles missing values, scales numerical features, engineers the "Overall Confidence" feature, removes outliers, handles class imbalance via SMOTE, and performs train/validation/test splits, with all steps clearly documented.
*   **Model Implementation:** The notebook implements and tunes three models: Random Forest, SVM, and Neural Network.  These are trained on the training data using `GridSearchCV`, tuning each model to optimize for F1 score. The code to run these, and the chosen parameters are all included in the notebook.
*   **Ensemble Model:** It constructs an ensemble model by combining the predictions of the tuned models using a hard-voting classifier.
*  **Evaluation:** The notebook shows that each of the models are evaluated using relevant performance metrics such as accuracy, precision, recall, F1 score and the confusion matrices, all of which are presented in a descriptive and organized way.

## Results

The notebook includes a section for detailed evaluation of the models, using both a confusion matrix and a table summarizing the scores for each model. In general, the Random Forest model showed superior performance, with the Ensemble model achieving good results, but not always exceeding those of a well-tuned Random Forest.

The notebook shows the results using the following format:
* A confusion matrix for each model, with a visualization.
* A summary table of model evaluation scores, showing Accuracy, Precision, Recall and F1-score.
* A bar plot of the Model performance, comparing each model's performance on the four evaluation metrics.

For specific results and observations, please refer to the notebook's output after execution.

## License

This project is licensed under the MIT License.

Summary of the MIT License:

Permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
Attribution must be given to the original authors.
Warranty Disclaimer: The software is provided "as is", without warranty of any kind. The authors are not liable for any damages arising from the use of the software.


Copyright (c) 2024 Sameh Shehata (https://github.com/samehshi).
