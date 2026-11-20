# Academic Integrity
Acknowledgemment use of the following external sources:
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


python3 k_nn.py image_train.csv --k 2 --metric cosine --normalization none --weights distance --test image_test.csv --output image_predictions.csv
generates: output image_predictions.csv

To view every available setting for a script, use --help:
python3 neural_network.py --help