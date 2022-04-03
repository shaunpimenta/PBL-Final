from flask import Flask, render_template,request
from mineralf import main
import os
import cv2
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for, render_template,Response
from werkzeug.utils import secure_filename
from keras import models
from mineralf import main
path = r'C:\Users\Home\Desktop\MyGallery\PBL FLask\static\predict'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
TransferLearningModel = models.load_model('PBL_MoonWalkers_TL_UNET.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app= Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PREDICTION_FOLDER = 'static/predict/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # print('upload_image filename: ' + filename)
        # file.save(os.path.join(app.config['PREDICTION_FOLDER'],filename+'_pred'))
        img = cv2.imread('./static/uploads/' + filename)
        print("Model file loaded")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 500))
        img = img.reshape(1, 500, 500, 3)
        print("Predicting")
        prediction = TransferLearningModel.predict(img)
        print("Prediction Done")
        pred = prediction.reshape(500, 500, 3)
        print(pred)
        # cv2.imshow('pred',pred)
        pred_ = cv2.resize(pred, (700, 450))
        plt.axis('off')
        plt.imshow(pred_)
        plt.savefig(f"static/predict/{filename}pred.png", facecolor='white', transparent=True, bbox_inches='tight')
        return render_template('upload.html',filename=filename)
    return None
@app.route('/predict',methods=['GET','POST'])
def predictm():
    # if request.method == 'POST':
    if __name__=='__main__':
        main()
    return render_template('predict.html')
if __name__=="__main__":
    # main()
    app.run(debug=True)