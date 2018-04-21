from utilities import get_data, display_metrics

from sklearn.svm import LinearSVC as SVC

x_train, x_test, y_train, y_test = get_data()
clf = SVC(multi_class='ovr')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
display_metrics(y_pred, y_test)
