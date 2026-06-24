from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar todos los artefactos del modelo entrenado
modelo = joblib.load('modelo.pkl')
scaler = joblib.load('scaler.pkl')
imputer_num = joblib.load('imputer_num.pkl')
imputer_cat = joblib.load('imputer_cat.pkl')
precio_mediano_marca = joblib.load('precio_mediano_marca.pkl')
mediana_global = joblib.load('mediana_global.pkl')

MARCAS_DISPONIBLES = sorted(precio_mediano_marca.index.tolist())


@app.route('/', methods=['GET'])
def inicio():
    return render_template('index.html', marcas=MARCAS_DISPONIBLES, resultado=None, error=None)


@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        marca = request.form.get('brand')
        milage = float(request.form.get('milage'))
        engine_hp = request.form.get('engine_hp')
        engine_litros = request.form.get('engine_litros')
        model_year = int(request.form.get('model_year'))
        tuvo_accidente = int(request.form.get('tuvo_accidente'))

        if milage < 0:
            raise ValueError('El kilometraje no puede ser negativo.')
        if model_year < 1900 or model_year > 2030:
            raise ValueError('El año del modelo no es válido.')

        engine_hp = float(engine_hp) if engine_hp else np.nan
        engine_litros = float(engine_litros) if engine_litros else np.nan

        antiguedad = 2026 - model_year

        fila = np.array([[milage, engine_hp, engine_litros, antiguedad, tuvo_accidente]])
        fila_imputada = imputer_num.transform(fila)

        marca_valor = precio_mediano_marca.get(marca, mediana_global)

        fila_completa = np.array([[
            fila_imputada[0][0],  # milage
            fila_imputada[0][1],  # engine_hp
            fila_imputada[0][2],  # engine_litros
            marca_valor,          # marca_precio_medio
            fila_imputada[0][3],  # antiguedad
            fila_imputada[0][4],  # tuvo_accidente
        ]])

        fila_escalada = scaler.transform(fila_completa)
        prediccion = modelo.predict(fila_escalada)[0]
        prediccion = max(prediccion, 0)  # el precio nunca puede ser negativo

        return render_template('index.html', marcas=MARCAS_DISPONIBLES,
                                 resultado=round(prediccion, 2), error=None)

    except Exception as e:
        return render_template('index.html', marcas=MARCAS_DISPONIBLES,
                                 resultado=None, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
