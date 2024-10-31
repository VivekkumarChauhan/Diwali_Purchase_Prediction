import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from flask import Flask, request, jsonify, render_template
from utils import recommend_products  # This should now work

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the index page

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    customer_id = data.get('customer_id')
    recommendations = recommend_products(customer_id)
    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
