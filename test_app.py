"""
Minimal test version of CRYPTIX-ML for PythonAnywhere debugging
Use this if the main web_bot.py fails to import
"""

from flask import Flask, jsonify
import sys
import os

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>🚀 CRYPTIX-ML Test Page</h1>
    <h2>✅ Flask is working on PythonAnywhere!</h2>
    <p><strong>Python Version:</strong> {}</p>
    <p><strong>Working Directory:</strong> {}</p>
    <p><strong>Python Path:</strong></p>
    <ul>
    {}
    </ul>
    <h3>Next Steps:</h3>
    <ol>
    <li>If you see this page, Flask is working correctly</li>
    <li>Check that all your files are uploaded</li>
    <li>Install missing dependencies: <code>pip3.9 install --user -r requirements.txt</code></li>
    <li>Update WSGI file to import from web_bot instead of test_app</li>
    </ol>
    """.format(
        sys.version,
        os.getcwd(),
        '\n'.join([f'<li>{path}</li>' for path in sys.path[:10]])
    )

@app.route('/test')
def test():
    """Test endpoint to verify basic functionality"""
    return jsonify({
        'status': 'success',
        'message': 'CRYPTIX-ML Flask app is working!',
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'files_in_directory': os.listdir('.') if os.path.exists('.') else []
    })

@app.route('/check-imports')
def check_imports():
    """Check if required modules can be imported"""
    results = {}
    modules_to_test = [
        'flask', 'pandas', 'numpy', 'requests', 
        'python_binance', 'config', 'position_manager'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            results[module] = '✅ Available'
        except ImportError as e:
            results[module] = f'❌ Missing: {str(e)}'
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
