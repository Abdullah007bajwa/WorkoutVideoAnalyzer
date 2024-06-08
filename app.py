from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils.video_classification import classify_video
from models.model import load_best_model 
from main import train_dir
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            best_model = load_best_model('optimized_exercise_classifier_model.h5')
            class_dict = {class_name: idx for idx, class_name in enumerate(os.listdir(train_dir))}  # Update the path
            results = classify_video(filepath, best_model, class_dict)
            return render_template('results.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
