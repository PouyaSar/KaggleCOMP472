# Academic Integrity
Acknowledgement use of the following external sources:
AI tools (Gemini and ChatGPT)
https://nycdatascience.com/blog/student-works/machine-learning/knn-classifier-from-scratch-numpy-only/
https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/tree/_classes.py#L699
https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/ensemble/_forest.py#L1174
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html


# k-nearest neighbours
python3 k_nn.py image_train.csv --k 2 --metric cosine --normalization none --weights distance

# Decision tree
python3 decision_tree.py image_train.csv --max-depth 20 --criterion gini

# Random forest
python3 random_forest.py image_train.csv --n-estimators 50 --max-depth none

# Neural network
python3 neural_network.py sentiment_train.csv --normalization l2 --hidden-layers 256 64 --learning-rate 0.001 --epochs 100 --patience 15

# Hybrid model (k-NN + neural networks)
python3 knn_hybrid.py image_train.csv
prints: validation accuracy only, no files


python3 k_nn.py image_train.csv --k 2 --metric cosine --normalization none --weights distance --test image_test.csv --output image_predictions.csv
generates: output image_predictions.csv

python3 knn_hybrid.py image_train.csv --test image_test.csv --output image_predictions.csv
retrains on all labeled rows and predicts the test set
generates: output image_predictions.csv

# Reproducing the submission

copy the competition CSVs from the Kaggle Data tab into this folder

python3 neural_network.py sentiment_train.csv --normalization l2 --hidden-layers 256 64 --learning-rate 0.001 --epochs 100 --patience 15 --test sentiment_test.csv --output sentiment_predictions.csv

generates: output sentiment_predictions.csv

python3 knn_hybrid.py image_train.csv --test image_test.csv --output image_predictions.csv

generates: output image_predictions.csv

combine sentiment_predictions.csv and image_predictions.csv with the example submission code from the competition instructions

generates: output submission.csv
