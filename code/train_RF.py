from sklearn.ensemble import RandomForestClassifier
from utilities import get_data, display_metrics

x_train, x_test, y_train, y_test = get_data()
# For reproducibility
clf = RandomForestClassifier(n_estimators=30, criterion='entropy')
clf.fit(x_train, y_train)
display_metrics(clf.predict(x_test), y_test)
