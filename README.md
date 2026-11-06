# Machine Learning Assignment 2: Netflix Dataset

## a. Problem statement
The goal of this project is to build an end-to-end Machine Learning classification web application using Streamlit. The problem is a binary classification task to predict whether a given title on Netflix is a 'Movie' (0) or a 'TV Show' (1) based on various features such as duration, release year, description length, and ratings.

## b. Dataset description
The dataset used is the Netflix Titles dataset. We engineered 12 numerical and categorical features to meet the assignment requirements of having at least 12 features and 500 instances.
**Features (12 total):**
- `duration_num`: Extracted numerical duration.
- `has_director`: Binary flag if a director is listed.
- `has_cast`: Binary flag if cast is listed.
- `desc_length`: Character length of the description.
- `title_length`: Character length of the title.
- `num_categories`: Number of genres listed.
- `is_us`: Binary flag if country is United States.
- `is_india`: Binary flag if country is India.
- `rating_TV-MA`: Binary flag for TV-MA rating.
- `rating_TV-14`: Binary flag for TV-14 rating.
- `rating_R`: Binary flag for R rating.
- `release_year`: Year of release.
- `target`: 0 for Movie, 1 for TV Show.

## c. Github Repository Link
[Add your GitHub Link here]

## d. Models used
### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9977 | 1.0 | 0.9928 | 1.0 | 0.9964 | 0.9947 |
| Decision Tree | 0.9994 | 0.9991 | 1.0 | 0.9982 | 0.9991 | 0.9987 |
| kNN | 0.9824 | 0.9949 | 0.9796 | 0.9635 | 0.9715 | 0.9588 |
| Naive Bayes | 0.9949 | 1.0 | 0.9982 | 0.9854 | 0.9917 | 0.9881 |
| Support Vector Machine | 0.996 | 1.0 | 0.9909 | 0.9964 | 0.9936 | 0.9908 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a very strong baseline with excellent AUC and precision. |
| Decision Tree | Captures the split effectively, yielding extremely high accuracy. 'duration' behaves differently for movies/TV shows. |
| KNN | Performs well but is slightly sensitive to the scaling of description and title lengths. |
| Naive Bayes | Struggles slightly with feature dependencies but maintains a very high recall rate. |
| Support Vector Machine | Outstanding overall performance, maximizing margin effectively for this specific feature space. |
| Overall Winner | **Decision Tree / SVM**, since the engineered features cleanly distinguish Movies vs TV Shows, leading to near-perfect metrics. |
