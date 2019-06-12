from datas_file import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D
from sklearn import  linear_model, model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import numpy as np

def make_clean_data(file1, file2):

    data = Data(file1, file2)
    data.tri_code_postale(codes_postals)
    data.tri_columns()
    data.create_column_surface_price()
    data.delete_dirty_data()

    return data

def linear_regression(data):

    data_x_train = data.iloc[:, [2]][-7500:]
    data_x_test = data.iloc[:, [2]][:-7500]

    data_y_train = data.iloc[:, [0]][-7500:]
    data_y_test = data.iloc[:, [0]][:-7500]

    regr = linear_model.LinearRegression()

    regr.fit(data_x_train, data_y_train)

    data_y_pred = regr.predict(data_x_test)

    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(data_y_test, data_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred))
    print(regr.coef_)
    print(regr.intercept_)



    # Plot outputs
    plt.scatter(data_x_test, data_y_test, color='black')
    plt.plot(data_x_test, data_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def multi_linear_regression(data):

    X = data[['surface_reelle_bati', 'nombre_pieces_principales']]
    X = sm.add_constant(X)
    y = data['valeur_fonciere']

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(data["surface_reelle_bati"], data["nombre_pieces_principales"], data["valeur_fonciere"], c='r',
               marker='^')

    ax.set_xlabel('surface en metre carré')
    ax.set_ylabel('nb_pieces')
    ax.set_zlabel('prix en €')

    plt.show()

def decision_tree(data):

    X = data[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude']]
    y = data['valeur_fonciere']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.30, random_state=0)



    regr_1 = DecisionTreeRegressor(max_depth=8)
    regr_1.fit(X_train, y_train)
    y_pred = regr_1.predict(X_test)



    """pgrid = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid=pgrid, scoring='neg_mean_squared_error', cv=10)
    grid_search.fit(X_train, y_train)
    y_predicted = grid_search.best_estimator_.predict(X_test)
    print(mean_squared_error(y_test, y_predicted))
    print(grid_search.best_params_)"""

    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

    f = 1 - mean_absolute_error(y_test, y_pred) / y.mean()
    print('Fiabilité :', f)



