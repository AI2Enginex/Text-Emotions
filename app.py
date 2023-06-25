
import re
import pickle 
import numpy as np
import cleantext
from flask import render_template
from keras.models import load_model
from flask import Flask, request , flash


class Parameters:

    def __init__(self):

        self.tf_matrix = pickle.load(open('./tf_matrix.pkl','rb'))
        self.ann_model = load_model("emotions.h5")
        self.labels = ['anger','fear','joy','sadness']
        
class CleanData(Parameters):

    def __init__(self):

        super().__init__()
    
    def user_input(self,val):
        
        return [re.sub('[^a-zA-Z]',' ' , message) for message in val]
    
    def gentext(self,val):

        return [data.replace(',','').strip() for data in self.user_input(val)]
    
    def training_list(self,val):

        return [cleantext.clean(word,stemming=True,stopwords=True) for word in self.gentext(val)]


class PredictLabel(Parameters):

    def __init__(self):
        super().__init__()

        
    def get_data(self,val):

        cd = CleanData()
        return cd.training_list(val)

    def genetare_tfid_vector(self,val):
    
        return self.tf_matrix.transform(self.get_data(val)).toarray()

    def predictlabel(self,val):

        vectors = self.genetare_tfid_vector(val)
        return self.labels[np.argmax(self.ann_model.predict(vectors))]
    
class FlaskApp(Parameters):

    def __init__(self):

        super().__init__()
        self.app = Flask(__name__ , template_folder = 'templates')
        self.app.secret_key = 'batman'
        self.pr = PredictLabel()

    def home(self):

        return render_template("index.html")

    def predict(self):

        try:
            user_input = [request.form["comment"]]
            result = self.pr.predictlabel(val=user_input)
            return render_template('index.html',result=result)
        except Exception as e:

            flash(category="error",message=e)
            return render_template("index.html")



    def run_app(self):

        self.app.add_url_rule('/', methods=['GET','POST'] , view_func=self.home)
        self.app.add_url_rule('/predict', methods=['GET','POST'] , view_func=self.predict)
        self.app.run(debug=True)

    
   
if __name__ == '__main__':

    app = FlaskApp()
    app.run_app()

   