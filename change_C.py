

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# generate the data
pos_mean = [1, 1]
pos_cov = [[1, 0], [0, 1]]
pos_points = np.random.multivariate_normal(pos_mean, pos_cov, 1000)
pos_labels = [1]*1000

pos_test_points = np.random.multivariate_normal(pos_mean, pos_cov, 500)
pos_test_labels = [1]*500

neg_mean = [-1, -1]
neg_cov = [[3, 0], [0, 3]]
neg_points = np.random.multivariate_normal(neg_mean, neg_cov, 1000)
neg_labels = [-1]*1000

neg_test_points = np.random.multivariate_normal(neg_mean, neg_cov, 500)
neg_test_labels = [-1]*500

train_X = np.concatenate([pos_points, neg_points], axis=0)
train_y = pos_labels + neg_labels

test_X = np.concatenate([pos_test_points, neg_test_points], axis=0)
test_y = pos_test_labels + neg_test_labels
# train the classifier
C=[1, 10, 100, 1000]
g = 10

classifier = []
accuracy = []

for c in C:
    this_classifier = SVC(C=c, gamma=g, kernel="rbf")
    classifier.append(this_classifier.fit(train_X, train_y))
    accuracy.append(classifier[-1].score(test_X, test_y))
# this code comes straight from the sklearn website
h=0.1

X=test_X
y=test_y

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['$C=1$, accuracy= 85.9',
          '$C=10$, accuracy= 84.6',
          '$C=100$, accuracy= 83.1',
          '$C=1000$, accuracy= 81.3']

plt.figure(figsize=(15, 10))
for i, clf in enumerate(classifier):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.5)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.rainbow)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()