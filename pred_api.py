from flask import request, Flask, jsonify
from Immo_model import *

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def test():
    lat = float(request.args['latitude'])
    longi = float(request.args['longitude'])
    surface = float(request.args['superficie'])
    pieces = float(request.args['nb_pieces'])


    modele = Immo_model('valeurs-foncieres2017.csv', 'valeurs-foncieres2018.csv')
    modele.classification_tree_model()
    modele.regression_tree_model()
    modele.predict_cluster()

    res = modele.predict(surface, pieces, lat, longi)

    return jsonify({'estimation': res})


if __name__ == "__main__":
    app.run(debug=True)


