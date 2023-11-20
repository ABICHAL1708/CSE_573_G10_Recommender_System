from flask import Flask, render_template, request, jsonify
from process_id import process_id

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_id', methods=['POST'])
def process_input_route():
    input_value = request.form['inputValue']
    selected_option = request.form['selectedOption']
    result = process_id(input_value, selected_option)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
