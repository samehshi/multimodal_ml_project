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
- [Further Work](#further-work)
- [Contributing](#contributing)
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

**Note**: Due to data privacy concerns, ensure the data file (`Ten-students_combined_data.xlsx`) is **not committed** directly to the repository if not already done. Please use a sample dataset and not the real data.

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

The notebook (`Code.ipynb`) covers the following:

*   **Exploratory Data Analysis (EDA):** Visualizing distributions, relationships, and correlations in the data.
*   **Data Preprocessing:** Handling missing values, scaling numerical features, feature engineering, outlier removal, and class imbalance using SMOTE.
*   **Data Splitting:** Stratified splitting into training, validation, and test sets.
*   **Model Implementation:** Training and hyperparameter tuning of Random Forest, SVM, and MLP models using GridSearchCV, utilizing F1-score to determine optimal parameters.
*   **Ensemble Model:** Combining the trained models using a Voting Classifier.
*   **Model Evaluation:** Comprehensive evaluation of all models using accuracy, precision, recall, F1-score, and confusion matrices.

## Results

The notebook includes a section for detailed evaluation of the models, using both a confusion matrix and a table summarizing the scores for each model. In general, the Random Forest model showed superior performance, with the Ensemble model achieving good results, but not always exceeding those of a well-tuned Random Forest.

For specific results, please refer to the notebook's output after execution.

## Further Work

Future avenues for development may include:

*   Exploring more advanced deep learning architectures.
*   Feature selection or engineering using insights from feature importance.
*   Investigating different data augmentation techniques.
*   Testing the models with external validation sets.

## Contributing

Contributions are welcome! If you would like to contribute to the project, please:

1.  Fork the repository.
2.  Create a new branch for your feature or fix.
3.  Commit your changes.
4.  Open a pull request.

## License

This project is licensed under the MIT License.

Summary of the MIT License:

Permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
Attribution must be given to the original authors.
Warranty Disclaimer: The software is provided "as is", without warranty of any kind. The authors are not liable for any damages arising from the use of the software.


Copyright (c) 2024 Sameh Shehata (https://github.com/samehshi).
