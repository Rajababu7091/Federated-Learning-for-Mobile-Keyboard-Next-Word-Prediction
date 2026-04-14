"""
dev.py — Hot-reload development server
Run:  python dev.py
URL:  http://localhost:5000

Watches frontend/ for any file changes and auto-refreshes the browser.
Equivalent to `npm run dev` with HMR — no Node.js required.
"""

import os
import sys
from livereload import Server
from app import app
from backend.state import load_model_state

if __name__ == '__main__':
    print('\n  🔐  FedKeyboard — DEV server with hot-reload')
    print('  Watching: frontend/templates/, frontend/static/')
    print('  Loading vocabulary and model…')
    loaded = load_model_state()
    print('  [OK] Model loaded' if loaded else '  [!]  No model — click "Train Model" in UI')
    print('\n  ➜  Open http://localhost:5000  (auto-refreshes on file save)\n')

    server = Server(app.wsgi_app)

    # Watch all frontend files — browser refreshes on any change
    server.watch('frontend/templates/*.html')
    server.watch('frontend/static/css/*.css')
    server.watch('frontend/static/js/*.js')

    # Also restart server on backend changes
    server.watch('backend/**/*.py', delay=1)
    server.watch('app.py', delay=1)

    server.serve(host='0.0.0.0', port=5000, debug=True, open_url_delay=1)
