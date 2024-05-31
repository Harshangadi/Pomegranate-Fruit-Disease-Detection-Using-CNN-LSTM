from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import os
import shutil
from datetime import datetime

app = Flask(__name__)

# create a folder to store images
if not os.path.exists(os.path.join(os.getcwd(), 'uploads')):
    os.makedirs(os.path.join(os.getcwd(), 'uploads'))

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

path = os.path.join(os.getcwd(), "Model", "model.h5")
# For azure
    
# Download and save the model
# download_model(model_url, save_path)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL


# Load the trained model
model = load_model(path)

# Define the target image size
target_size = (256, 256)

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize the image

def detect_disease(file=""):
    print(file)
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    print(predictions)
    # Alternaria/  Anthracnose/  Bacterial_Blight/  Cercospora/  Healthy/
    class_labels = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']
    predicted_class = class_labels[class_index]
    result = {
        'detected_disease': predicted_class,
        'Alternaria': f"{(predictions[0][0]*100):.02f}",
        'Anthracnose': f"{(predictions[0][1]*100):.02f}",
        'Bacterial_Blight': f"{(predictions[0][2]*100):.02f}",
        'Cercospora': f"{(predictions[0][3]*100):.02f}",
        'Healthy': f"{(predictions[0][4]*100):.02f}"
    }
    return result

@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))


@app.route('/manifest.json')
def serve_manifest():
    return app.send_static_file('manifest.json', mimetype='application/manifest+json')


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')

@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent')
    user_agent = user_agent.lower()

    print(user_agent)

    if "android" in user_agent or "iphone" in user_agent:
        return render_template('mobile/detect.html')
    else:
        return render_template('desktop/index.html')


@app.route('/upload', methods=('POST',))
def upload():
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    files = request.files.getlist('files')
    for file in files:
        fn = secure_filename(file.filename)
        if fn != '':
            file_ext = os.path.splitext(fn)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return 'Invalid file type', 400
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fn))
            file_urls.append(fn)
    session['file_urls'] = file_urls
    return "ok" # change to your own logic
    # return redirect(url_for('results')) # change to your own logic

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/results')
def results(file_urls=None):
    user_agent = request.headers.get('User-Agent')
    user_agent = user_agent.lower()
    print(user_agent)
    # set the file_urls and remove the session variable
    if "file_urls" not in session or session['file_urls'] == []:
        file = request.args.get('filename')
        # check if the file is present in the upload folder
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file)):
            # copy file from static/public folder to uploads folder using shutil
            shutil.copy(os.path.join(os.getcwd(), 'static', 'public', file), os.path.join(app.config['UPLOAD_FOLDER'], file))
        file_urls = [file]
    else:
        file_urls = session.pop('file_urls', [])
    # file_urls = ["IMG_20230813_151923.jpg", "IMG_20230910_102304.jpg"]
    print(file_urls)
    dis_results = []
    for file in file_urls:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        res = detect_disease(file_path)
        file_stat = os.stat(file_path)
        upload_time = datetime.fromtimestamp(file_stat.st_mtime)
        res['filename'] = file
        print(res['filename'])
        res["upload_time"] = upload_time
        res["size"] = file_stat.st_size
        dis_results.append(res)

    if "android" in user_agent or "iphone" in user_agent:
        print("Mobile")
        return render_template('mobile/results.html', results=dis_results)
    else:
        return render_template('desktop/results.html', results=dis_results)
    # return render_template('results.html', file_urls=file_urls)

@app.route('/detect')
def detect():
    return render_template('desktop/detect.html')

@app.route('/info')
def info():
    return render_template('desktop/hifi-under-development.html')

@app.route('/help')
def help():
    return render_template('desktop/hifi-help.html')

if __name__ == '__main__':
    # app.run(host='192.168.175.193', debug=True)
    app.run(debug=True)


