# Adversarial-Debiasing

This repository summarizes an implementation of Adversarial Debiasing in PyTorch to address the problem of Gender Bias in toxic comment classification. This is the outcome of a class project completed by students at the University of Pennsylvania.

### Problem Statement

This dataset and problem statement were sourced from the Jigsaw Unintended Bias in Toxicity Classification competition on Kaggle. Due to the overrepresentation of certain identity groups (gender, racial, sexual identities) in toxic comments, regular toxicity classifiers suffer from unintended bias - they overpredict toxicity for certain groups versus others. This project is an attempt to implement the technique of Adversarial Debiasing to train a "fair" classifier that does not suffer from the particular problem of gender bias. The objectives of this project were threefold:
- Extend the limited implementations of Adversarial Debiasing to a new domain of Toxicity Classification
- Examine the fairness-performance tradeoff faced by our model, an issue that has not been commented on in much detail within previous implementations
- Test the generalizability of our debiased model on a dataset from a different domain, that it has not been trained on

### Adversarial Debiasing

In 2018, Zhang et. al. [1] proposed adversarial networks as a technique for fighting model bias. This was a variation on generative adversarial networks proposed by Goodfellow, et. al [2]. The framework proposed involved the generator learning with respect to a protected attribute, like gender. This translated into a structure where the generator prevents the discriminator from being able to predict gender under a given overarching task. In their paper, Zhang et. al.[1] were able to demonstrate an improvement in fairness on an income classification task, using their Adversarial Debiasing approach, facing only a 1.5% compromise in overall accuracy. This process of Adversarial Debiasing can be generalized to any setting where the model uses a gradient based learning including both regression and classification tasks and is hence, suitable for our task at hand.

[1] Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 2018. Mitigating Unwanted Biases with Adversarial Learning. In AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society (AIES‘18).
[2] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems (NIPS‘14).

### Experiments

We utilized three models for our experiments, to ascertain the success of our Adversarial Debiasing approach:

- Baseline XGBoost Classifier: To get a baseline understanding of how well a strong simple classifier would perform on our task of toxicity classification, we utilized the distributed gradient boosting library XGBoost. This was implemented on top of a TF-IDF based bag of words representation of our comment text. This allowed us to have a benchmark, against which we could measure the performance of our regular classifier, thus ensuring we built a strong classifier to begin with. Implementation can be found here: [BoW_Toxicity_Classifier.ipynb](https://github.com/choprashweta/Adversarial-Debiasing/blob/master/BoW_Toxicity_Classifier.ipynb)

- Classifier: Our regular classifier was built in the form of a neural network. We utilized an uncased pretrained BERT layer to form our comment text representations, using a PyTorch implementation of Google Research’s BERT model, created by HuggingFace. On top of this, we added one dropout and three linear layers to create the Classifier. Implementation can be found here: [Regular_Classifier.ipynb](https://github.com/choprashweta/Adversarial-Debiasing/blob/master/Regular_Classifier.ipynb)

- Fair Classifier (Classifier + Adversary): For our final model that implements the Adversarial Debiasing, we utilized our regular classifier, itself, as the predictor portion of the model. For the adversary, we utilized a distinct shallow network of two linear layers. The point of connection between the two networks, the classifier and the adversary, was that the hidden penultimate layer of the classifier formed the input to the adversary. Implementation can be found here: [Debiased_Classifier.ipynb](https://github.com/choprashweta/Adversarial-Debiasing/blob/master/Debiased_Classifier.ipynb)

Running our data through these three models, we calculated both performance and fairness metrics on a held-out test dataset to report our results.

### Performance and Fairness Metrics

Evaluation of both performance and fairness form an important part of our inquiry. 

Performance: Given the imbalanced distribution of toxicity within our dataset, we utilized the F1-Score as the primary measure of model performance. Accuracy has also been reported but must be read while keeping in mind the distribution of toxic and non-toxic comments in our dataset. During model training precision and recall were also separately monitored to ensure a relatively balanced F1-Score representation.

Fairness: Rising interest in the field of algorithmic fairness has brought about several definitions of how model fairness can be measured. Here, we move forward with three commonly used measures, defined below in the context of our implementation [1]:

- Demographic Parity: A predictor satisfies Demographic Parity if its prediction is independent of the protected class. In our case of gender being the protected class, and toxicity being the target class variable, this can be understood using the following simplification: P[Y=Toxic] = P[Y=Toxic| Male] = P[Y=Toxic| Female]. Thus, the rate of toxicity prediction for both genders should be close to equal.
- Equality of Opportunity (True Positive Parity): A predictor satisfies Equality of Opportunity with respect to a class y, if the probability of a true prediction is independent of the protected class, conditioned on the true target class being y. In our case of gender being the protected class, and toxicity being the target class variable, this can be understood using the following simplification: P(Prediction = Toxic | Y = Toxic) = P(Prediction = Toxic| Male, Y = Toxic) = P(Prediction = Toxic| Female, Y = Toxic). Thus, the rate of true positives for both genders should be close to equal.
- False Positive Parity: A predictor satisfies False Positive Parity with respect to a class y, if the probability of a false prediction is independent of the protected class, conditioned on the true target class being y. In our case of gender being the protected class, and toxicity being the target class variable, this can be understood using the following simplification: P(Prediction = Toxic | Y = Non-Toxic) = P(Prediction = Toxic| Male, Y = Non-Toxic) = P(Prediction = Toxic| Female, Y = Non-Toxic). Thus, the rate of false positives for both genders should be close to equal.

Since these fairness measures are boolean (True-False) in nature, we implement them, instead, in the form of differences in probabilities for the Female and Male protected classes. This allows us to utilize a continuous measure of difference between scores for both genders which we aim to reduce to zero through our adversarial debiasing approach.

### Summary of Results

The Regular classifier demonstrated bias across all three of the fairness metrics we examined. As compared to the Regular Classifier, the Fair Classifier (Classifier + Adversary) had a significant improvement in fairness across all the three fairness metrics we had chosen. Our Fair Classifier was able to comfortably beat the baseline XGBoost Model, however caused a small drop in performance over the Regular Classifier - demonstrating the tradeoff between performance and fairness.

The Fair Classifier we trained generalized well to a new dataset from a different data source and domain as well. 

The Adversarial training process itself was challenging - particularly the tuning of the fairness-performance tradeoff, structuring of the training cycles and the sharing of information across the predictor and adversary portions of the model.

Details of the implementation and results can be found in our full project report [Project Report](https://github.com/choprashweta/Adversarial-Debiasing/blob/master/CIS_519_Project_Report%20(4).pdf). In case of questions feel free to reach out at shweta.shwetachopra@gmail.com

