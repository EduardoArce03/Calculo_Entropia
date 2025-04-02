from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    with open('test.json') as f:
        data = json.load(f)

    return render_template('/index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
