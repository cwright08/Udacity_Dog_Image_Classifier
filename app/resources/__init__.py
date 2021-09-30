from flask import Flask

app = Flask(__name__)
app.secret_key = "secret key"
UPLOAD_FOLDER = app.root_path+"/static/images/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
print(UPLOAD_FOLDER)
print(app.static_folder)
print(app.config['UPLOAD_FOLDER'])