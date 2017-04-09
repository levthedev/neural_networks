from flask import Flask, render_template, request
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
    # np.set_printoptions(linewidth=150, precision=1)
    net = load('./saved')
    vector_result = net.feedforward(clean_input)
    result = np.argmax(vector_result)
    # pdb.set_trace()
    return str(result.tolist())

if __name__ == "__main__":
    app.run(debug=True)
