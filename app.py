"""
app.py  —  Entry point
Run:  python app.py
URL:  http://localhost:5000
"""

import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from backend.state import load_model_state
from backend.routes.predict_routes     import predict_bp
from backend.routes.train_routes       import train_bp
from backend.routes.audit_routes       import audit_bp
from backend.routes.personalize_routes import personalize_bp

app = Flask(
    __name__,
    template_folder=os.path.join('frontend', 'templates'),
    static_folder=os.path.join('frontend', 'static'),
)
CORS(app)

app.register_blueprint(predict_bp)
app.register_blueprint(train_bp)
app.register_blueprint(audit_bp)
app.register_blueprint(personalize_bp)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500


if __name__ == '__main__':
    print('\n  FedKeyboard - starting server')
    print('  Loading vocabulary and model...')
    loaded = load_model_state()
    if loaded:
        print('  [OK] Model loaded from checkpoint')
    else:
        print('  [!]  No trained model found -- click "Train Model" in the UI')
    print('\n  Open http://localhost:3000\n')
    app.run(host='0.0.0.0', port=3000, debug=False)
