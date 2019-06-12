from datas_file import *
from sklearn import  model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


class Immo_model:

    def __init__(self, file1, file2):

        data = Data(file1, file2)
        data.tri_code_postale(codes_postals)
        data.tri_columns()
        data.create_column_surface_price()
        data.delete_dirty_data()
        data.create_cluster()

        self.data = data
        self.regr = None
        self.clf = None

    def classification_tree_model(self):

        X = self.data.df[['latitude', 'longitude']]
        y = self.data.df.cluster

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

        clf = DecisionTreeClassifier(splitter='best', random_state=1)
        clf = clf.fit(X_train, y_train)


        self.clf = clf

    def regression_tree_model(self):

        X = self.data.df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'cluster']]
        y = self.data.df['valeur_fonciere']

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.30, random_state=0)

        regr = DecisionTreeRegressor(criterion='friedman_mse', random_state=1, max_depth=8)
        regr.fit(X_train, y_train)

        self.regr = regr

    def predict_cluster(self):

        clust_pred = []

        for elt in self.clf.predict_proba(self.data.df[['latitude', 'longitude']]):
            max = 0
            index = 0
            for i, clust in enumerate(elt):
                if clust > max:
                    max = clust
                    index = i
            clust_pred.append(index + 1)

        self.data.df['cluster_pred'] = clust_pred

    def pred_clus(self, lat, lon):

        index = 0
        for elt in self.clf.predict_proba([[lat, lon]]):
            max = 0
            for i, clust in enumerate(elt):
                if clust > max:
                    max = clust
                    index = i
        return index + 1



    def error_of_model(self):

        X = self.data.df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'cluster_pred']]
        y = self.data.df['valeur_fonciere']

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.99, random_state=0)

        y_pred = self.regr.predict(X_test)

        print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

        print('Variance :', r2_score(y_test, y_pred))

    def predict(self, superficie, nb_pieces, latitude, longitude):

        cluster = self.pred_clus(latitude, longitude)
        res = self.regr.predict([[superficie, nb_pieces, latitude, longitude, cluster]])

        return res


