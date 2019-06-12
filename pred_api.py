# from flask import request
# from flask_api import FlaskAPI
# from math import pi

# app = FlaskAPI(__name__)


# @app.route("/", methods=['GET', 'POST'])
# def test():
#     lat = float(request.args[''])
#     longi = float(request.args[''])
#     surface = float(request.args[''])
#     pieces = float(request.args[''])

#     return {'': pi * rad**2}



# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

