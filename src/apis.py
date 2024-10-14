from flask import Flask
import os
from subprocess import Popen
import json

app = Flask(__name__)

@app.route("/line-violation")
def line_violation():
	start_process = Popen(['python', 'Project-GUI.py'])
	start_process.wait()
	return json.dumps({})

@app.route("/helmet-detection")
def helmet_detection():
	start_process = Popen(['python', 'helmet.py'])
	start_process.wait()
	return json.dumps({})

@app.route("/licence-plate-detection")
def licence__plate_recognition():
	start_process = Popen(['python', 'licence_plate.py'])
	start_process.wait()
	return json.dumps({})

@app.route("/automation")
def automation():
	start_process = Popen(['python', 'predictor.py'])
	start_process.wait()
	return json.dumps({})

    
if(__name__ == "__main__"):
    app.run(debug = True, port = '8000')