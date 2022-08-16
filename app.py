import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")

@app.route('/minor')
def minor():
  
    return render_template("minor.html")

@app.route('/major')
def major():
  
    return render_template("major.html")

@app.route('/gallery')
def gallery():
  
    return render_template("gallery.html")

@app.route('/contact')
def contact():
  
    return render_template("contact.html")
  
@app.route('/resume')
def resume():
  
    return render_template("resume.html")

@app.route('/decision')
def model1():
  
    return render_template("decision.html")

@app.route('/logistic')
def model2():
  
    return render_template("logistic.html")

@app.route('/svm')
def model3():
  
    return render_template("svm.html")

@app.route('/random')
def model4():
  
    return render_template("random.html")

@app.route('/knn')
def model5():
  
    return render_template("knn.html")

@app.route('/naive')
def model6():
  
    return render_template("naive.html")

@app.route('/kmeans')
def model7():
  
    return render_template("kmeans.html")


@app.route('/predict_decision',methods=['GET'])
def predict1():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('decision_model.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('decision.html', prediction_text='{}'.format(message))


@app.route('/predict_logistic',methods=['GET'])
def predict2():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('log_reg.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('logistic.html', prediction_text='{}'.format(message))

@app.route('/predict_svm',methods=['GET'])
def predict3():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('svm.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('svm.html', prediction_text='{}'.format(message))

@app.route('/predict_random',methods=['GET'])
def predict4():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('random_forest.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('random.html', prediction_text='{}'.format(message))

@app.route('/predict_knn',methods=['GET'])
def predict5():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('knn.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('knn.html', prediction_text='{}'.format(message))

@app.route('/predict_naive',methods=['GET'])
def predict6():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('naive.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('naive.html', prediction_text='{}'.format(message))

@app.route('/predict_kmeans',methods=['GET'])
def predict7():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    criminal=float(request.args.get('criminal'))
    assets=float(request.args.get('assets'))
    liability=float(request.args.get('liability'))
    p_votes=float(request.args.get('p_votes'))
    education=float(request.args.get('education'))
    gender=float(request.args.get('gender'))
    g_votes=float(request.args.get('g_votes'))
    t_votes=float(request.args.get('t_votes'))
    category=float(request.args.get('category'))
    over_t_votes=float(request.args.get('over_t_votes'))
    over_t_electors=float(request.args.get('over_t_electors'))
    t_electors=float(request.args.get('t_electors'))

   
    model=pickle.load(open('kmeans.pkl','rb'))
      

    dataset= pd.read_excel('election1.xlsx')
    X = dataset.iloc[:, 5:18]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[gender,criminal,age,category,education,assets,liability,g_votes,p_votes,t_votes,over_t_electors,over_t_votes,t_electors]]))
    if prediction==0:
      message="Election lost"
    elif prediction==1:
      message="Election Won"
    
        
    return render_template('kmeans.html', prediction_text='{}'.format(message))


if __name__ == "__main__":
    app.run(debug=True)
