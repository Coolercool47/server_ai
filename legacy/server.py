from flask import Flask, request, jsonify
import requests
from your_model import predict_is_ai
from flask_cors import CORS
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/api/check-ai', methods=['POST'])
def check_ai():
    try:
        data = request.get_json()
        image_data_url = data.get('imageData') # Изменить ключ на imageData
        if not image_data_url:
            return jsonify({"error": "No imageData provided"}), 400

        # Извлечь base64 данные из Data URL
        # Формат Data URL: data:[<mediatype>][;base64],<data>
        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded) # Декодировать base64

        # response = requests.get(image_url, timeout=5) # Эта часть больше не нужна
        # if response.status_code != 200:
        #     return jsonify({"error": "Failed to download image"}), 400
        # image_bytes = response.content # Используем декодированные байты

        result = predict_is_ai(image_bytes) # Вызываем твою модель с байтами
        return jsonify({"result": result})
    except Exception as e:
        # Добавим вывод ошибки в консоль сервера для отладки
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)