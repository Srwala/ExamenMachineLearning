from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar modelos
modelo_reg = joblib.load('checkpoints/modelo_regresion.pkl')
modelo_clf_data = joblib.load('checkpoints/modelo_clasificacion.pkl')
modelo_clf = modelo_clf_data['model']
scaler_clf = modelo_clf_data['scaler']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Datos del formulario
    data = request.form
    
    # --- Predicción de Regresión (TimeAlive) ---
    distancia = float(data['distancia'])
    kills = float(data['kills'])
    tiempo_pred = modelo_reg.predict([[distancia, kills]])[0]*1000
    
    # --- Predicción de Clasificación (RoundWinner) ---
    time_alive = float(data['time_alive'])
    survived = int(data['survived'])
    resultado = modelo_clf.predict(scaler_clf.transform([[time_alive, survived]]))[0]
    resultado_texto = "Gana" if resultado == 1 else "Pierde"
    
    return render_template('result.html', 
                         tiempo_pred=round(tiempo_pred, 1),
                         resultado=resultado_texto,
                         distancia=distancia,
                         kills=kills,
                         time_alive=time_alive,
                         survived=survived)

if __name__ == '__main__':
    app.run(debug=True)
