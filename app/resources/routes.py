from resources import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
import os

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

@app.route('/display/<filename>')
def display_image(filename):
    print(app.config['UPLOAD_FOLDER'])
    print(url_for(app.config['UPLOAD_FOLDER'],+filename))
    return redirect(url_for('static','/images/'+filename), code=301)

#Source https://roytuts.com/upload-and-display-image-using-python-flask/