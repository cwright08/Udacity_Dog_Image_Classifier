from resources import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from resources import model_prediction


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/upload_image", methods=[ "POST","GET"])
def upload_image():

	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = file.filename
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('index.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

# Display image uploaded 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename='images/'+filename), code=301)

#Source https://roytuts.com/upload-and-display-image-using-python-flask/

# Run Dog Detector Algorithm
@app.route('/go/<filename>')
def dog_algo(filename):
	from resources.model_prediction import dog_algo
	message = dog_algo(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	return render_template('go.html',predictsuccess = True, message=message, filename = filename) 