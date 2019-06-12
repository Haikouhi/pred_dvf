from flask import request
from flask_api import FlaskAPI
from Immo_model import *

app = FlaskAPI(__name__)

modele = Immo_model('valeurs-foncieres2017.csv', 'valeurs-foncieres2018.csv')
modele.classification_tree_model()
modele.regression_tree_model()
modele.predict_cluster()


@app.route("/", methods=['GET', 'POST'])
def test():
    lat = float(request.args['latitude'])
    longi = float(request.args['longitude'])
    surface = float(request.args['superficie'])
    pieces = float(request.args['nb_pieces'])

    res = modele.predict(surface, pieces, lat, longi)

    return {'estimation': res}


if __name__ == "__main__":
    app.run(debug=True)


