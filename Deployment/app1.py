from flask import Flask

app = Flask(__name__)

@app.route("/")
def func1():
    return "Hello world from Flask"

@app.route("/ml")
def func2():
    return "Hi, How are you doing today?"

if __name__=="__main__":
    app.run(debug=False,port=8000,host="0.0.0.0")