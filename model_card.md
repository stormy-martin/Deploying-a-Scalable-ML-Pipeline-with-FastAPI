# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a binary classifier using a scikit-learn RandomForestClassifier with 100 estimators, predicting whether income exceeds $50,000 from 1994 U.S. Census data. Categorical features are one-hot encoded, and the model is served via FastAPI.

## Intended Use
This model is an educational demonstration of an ML deployment pipeline. It should not be used for real-world decisions about individuals.

## Training Data
The model was trained on 80% of the UCI Census Income dataset, which is 26,048 rows with 14 features. The classes are imbalanced, with 75.9% of records labeled <=50K and 24.1% labeled >50K.

## Evaluation Data
The remaining 20% of the dataset, or 6,513 rows, was held out for testing. This test set was used for both the overall evaluation and the slice analysis.

## Metrics
The model was evaluated using precision, recall, and F1, which are more informative than accuracy because the classes are imbalanced. On the test set, it achieved a precision of 0.7327, a recall of 0.6397, and an F1 of 0.6830.

## Ethical Considerations
The model uses race and sex as features, reflecting economic inequalities in 1994. Performance varies by demographic group, with F1 dropping from 0.6855 for White records to 0.5333 for Amer-Indian-Eskimo records.

## Caveats and Recommendations
The data is from 1994 and does not reflect current income patterns. Hyperparameter tuning and addressing class imbalance would improve performance. This model should not be used for decisions about individuals.