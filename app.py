import os 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)

# Définir le dossier de téléchargement des images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Vérifier si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Faire la prédiction avec la fonction predict_image()
        predicted_class, confidence = predict_image(filepath)
        
        # Afficher les résultats
        return render_template('result.html', image_name=filename, predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)



