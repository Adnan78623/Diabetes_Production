from flask import Flask , render_template , request

import model

app = Flask(__name__)

dp = 0

@app.route("/", methods = ['GET','POST'])
def diabet():
    global dp
    if request.method == "POST":
        glc = request.form['glc']
        bp = request.form['bp']
        bmi = request.form['bmi']
        age = request.form['age']

        diabetes_pred = model.diabetes_perdiction(glc,bp,bmi,age)
        dp = diabetes_pred
    return render_template("index.html", dp = dp)



# @app.route("/sub", methods= ['POST'])
# def submit():
#     if request.method == "POST":
#         name = request.form["username"]
#     return render_template("sub.html", n = name)




if __name__ == "__main__":
    app.run(debug=True)