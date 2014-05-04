import scipy.io
import numpy as np
mat = scipy.io.loadmat('ex3data1.mat') # dataset from Andrew Ng ML class
import pylab as pl
from sklearn import linear_model, datasets,svm, metrics
#for index, (label, image) in enumerate(zip(mat['y'], mat['X'])[:4]):
#    print index
#    print label
#    print image


# The digits dataset
digits = datasets.load_digits()  # dataset from scikit-learn

# We have 8X8 pixel images of handwritten digits from 0-9, together with their labels
# We'll visualise the first 4

for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index + 1)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Training: %i' % label)

# We wish to compare the performance of logistic regression and SVM on this data set,
# but first we need to flatten the image, to represent the data as a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a SVM classifier and a logistic regression classifier
svm = svm.SVC(gamma=0.001)
logistic = linear_model.LogisticRegression(C=0.05)

# We learn the digits on the first half of the digits
svm.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
logistic.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
svmPredicted = svm.predict(data[n_samples / 2:])
logisticPredicted = logistic.predict(data[n_samples / 2:])

# Compare the performance of the SVM and Logistic classifiers
print "SVM classification report %s:\n%s\n" % (
    svm, metrics.classification_report(expected, svmPredicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, svmPredicted)

print "Logistic regression classification report %s:\n%s\n" % (
    logistic, metrics.classification_report(expected, logisticPredicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, logisticPredicted)

# We see that SVM outperformed logistic regression

# Visualise the SVM predictions
for index, (image, prediction) in enumerate(
    zip(digits.images[n_samples / 2:], svmPredicted)[:4]):
    pl.subplot(2, 4, index + 5)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Prediction: %i' % prediction)

pl.show()
