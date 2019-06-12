from flask import request
from flask_api import FlaskAPI
from math import pi

app = FlaskAPI(__name__)


@app.route("/", methods=['GET', 'POST'])
def test():
    lat = float(request.args['lattiude'])
    longi = float(request.args['longitude'])
    surface = float(request.args['surface_reelle_bati'])
    pieces = float(request.args['nombre_pieces_principales'])

    return {'': pi * rad**2}


if __name__ == "__main__":
    app.run(debug=True)

test 