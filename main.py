from functions import *

data = make_clean_data('valeurs-foncieres2017.csv', 'valeurs-foncieres2018.csv')
#linear_regression(data.df)
#multi_linear_regression(data.df)
decision_tree(data.df)