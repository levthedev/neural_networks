from flask import Flask, render_template, request, jsonify
import numpy as np
from improved_network import *
import pdb

app = Flask(__name__)
app.debug = True

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/process", methods=["GET"])
def process():
    dirty_input = request.args['vector'].encode('latin_1').split(',')
    clean_input = np.reshape(np.array(map(float, dirty_input)), (784, 1))
    net = load('./saved3')
    vector_result = np.hstack(net.feedforward(clean_input))
    probabilities = zip(range(0, 10), vector_result.tolist())
    scaled = [(t[0], "{0:.2f}".format(round(t[1] * 100, 2))) for t in probabilities]
    result = sorted(scaled, key=lambda tup: tup[1], reverse=True)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
