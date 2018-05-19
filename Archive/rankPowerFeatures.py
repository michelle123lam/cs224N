"""
rankPowerFeatures.py
-------
Aggregate all subordinate-to-superior emails and superior-to-subordinate emails to identify most relevant features for power

Formulations:
- [Grouped] email_contents_grouped.npy
- [Per-email] email_contents_nodup.npy

Grouped features:
- avg_num_recipients_grouped
- avg_num_tokens_per_email_grouped (un-bucketed amounts)

Per-email features:
- num_recipients_features_nodup
- gender_features_nodup

gender_map (from UID to gender)
	1 = Female; 2 = Male

# dominant sender, subordinate recipient = label 0
# subordinate sender, dominant recipient = label 1

"""

"""
Additional features to test out:
- average sentence length
- aggressiveness/toxicity?
- sentiment
- other measures of terseness
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from processData import batch_iter, load_data_and_labels, load_data_and_labels_bow

def main():
	# Investigate GROUPED version
	# Use features grouped by (sender, recipient) pairs
	email_contents_file = "email_contents_grouped.npy"
	labels_file = "labels_grouped.npy"
	emails, labels = load_data_and_labels(email_contents_file, labels_file)
	emails = np.array(emails)

	# Aggregate GROUPED features
	num_recipients_features = np.array(np.load("avg_num_recipients_grouped.npy"))
	num_words_features_unprocessed = np.array(np.load("avg_num_tokens_per_email_grouped.npy"))
	print "feature shapes:", np.shape(num_recipients_features), np.shape(num_words_features_unprocessed)
	agg_features = np.dstack((num_recipients_features, num_words_features_unprocessed))
	agg_features = agg_features[0]
	print "aggregated shape:", np.shape(agg_features)

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
	# emails_shuffled = emails[shuffle_indices]
	agg_features_shuffled = agg_features[shuffle_indices]
	labels_shuffled = labels[shuffle_indices]
	
	# Split data into train (0.6), dev (0.2), and test (0.2)
	train = 0.6
	dev_and_test = 0.4
	trainFeatures, devAndTestFeatures, trainLabels, devAndTestLabels = train_test_split(agg_features_shuffled, labels_shuffled, test_size=dev_and_test, random_state=42)
	test = 0.5
	devFeatures, testFeatures, devLabels, testLabels = train_test_split(devAndTestFeatures, devAndTestLabels, test_size=test, random_state=42)
	# TODO: save one particular train/dev/test split and reuse it

	# Perform PCA (Principal Component Analysis)
	pca = PCA(n_components=1)
	pca.fit(trainFeatures)
	print "PCA explained_variance_ratio_", pca.explained_variance_ratio_
	print "PCA singular_values_", pca.singular_values_

	# Perform Univariate Selection
	"""
	# feature extraction
	test = SelectKBest(score_func=chi2, k=4)
	fit = test.fit(X, Y)
	# summarize scores
	numpy.set_printoptions(precision=3)
	print(fit.scores_)
	features = fit.transform(X)
	# summarize selected features
	print(features[0:5,:])
	"""

	# Perform Recursive Feature Elimination
	"""
	model = LogisticRegression()
	rfe = RFE(model, 3)
	fit = rfe.fit(X, Y)
	print("Num Features: %d") % fit.n_features_
	print("Selected Features: %s") % fit.support_
	print("Feature Ranking: %s") % fit.ranking_
	"""

	# Perform Feature Importance with Extra Trees Classifier
	"""
	# feature extraction
	model = ExtraTreesClassifier()
	model.fit(X, Y)
	print(model.feature_importances_)
	"""



	# Aggregate low2hi and hi2low groups
	# Get features for each group
	# Perform feature selection for each group
	# Display features ranked by importance

if __name__ == "__main__":
	main()
