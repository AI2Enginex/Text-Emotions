
import re
import spacy
from spacy import displacy
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from text_package.Text import PhraseAnalysis

class LoadFile:

    def __init__(self,file_name):

        self.file = pd.read_csv(file_name)
        self.ner = spacy.load('en_core_web_lg')

class GenerateData(LoadFile):

    def __init__(self, file_name):
        super().__init__(file_name)

    def process_data(self,column_name):

       return [re.sub('[^a-zA-Z]',' ' , message)  for message in self.file[column_name]]
    
    def get_clean_text(self,column_name):

        return [' '.join(text.split()) for text in self.process_data(column_name)]
    
class NameEntityRecognition:

    def __init__(self,file_name):

        self.gd = GenerateData(file_name)
        self.phrase = PhraseAnalysis(base_file=file_name)
        self.ner = spacy.load('en_core_web_lg')


    def get_data(self,column_name):

        data_str = ' '.join(self.gd.get_clean_text(column_name))
        return data_str

    
    def namentity(self,column_name,file,plotype):

        text_str = self.ner(self.get_data(column_name))

        svj = displacy.render(text_str,style=plotype,page=True)

        with open(file,'w+',encoding='utf-8') as f:

            f.write(svj)

        f.close()
    

class WordNER(LoadFile):

    def __init__(self, file_name):

        super().__init__(file_name)

        self.class_obj = PhraseAnalysis(base_file=file_name)

    def get_dataframe(self,feature_name,word):

        return self.class_obj.filter_sentence(column_name=feature_name,word=word)
    
    def generate_list(self,feature_name,word):

        df = self.get_dataframe(feature_name=feature_name,word=word)
        return [re.sub('[^a-zA-Z]',' ' , message)  for message in df[feature_name]]
    
    def get_clean_text(self,feature_name , word):

        return [' '.join(text.split()) for text in self.generate_list(feature_name,word)]
    
class GetNameEntity(WordNER):

    def __init__(self,file):

        super().__init__(file_name=file)

        self.word = WordNER(file_name=file)

    def get_data(self,feature_name,word):

        return self.word.get_clean_text(feature_name,word)
    
    def gettextarr(self,column_name,word):

        return ''.join(self.get_data(feature_name=column_name,word=word))

    def namentity(self,column_name,word,file,plotype):

        text_str = self.ner(self.gettextarr(column_name,word))

        svj = displacy.render(text_str,style=plotype,page=True)

        with open(file,'w+',encoding='utf-8') as f:

            f.write(svj)

        f.close()
    
class WordGUI:

    def __init__(self):

        self.inner_canvas = tk.Tk()
        self.inner_canvas.geometry('400x250')
        self.inner_canvas.eval('tk::PlaceWindow . center')
        self.wordcanvas = tk.Canvas(self.inner_canvas, width=600, height=300)

        tk.Label(self.inner_canvas, text="Select CSV file",
             font=("Helvetica", 14)).place(x=80, y=20)
    
        tk.Label(self.inner_canvas, text="Feature Name",
             font=("Helvetica", 14)).place(x=80, y=55)
        
        tk.Label(self.inner_canvas, text="word to Filter",
             font=("Helvetica", 14)).place(x=80, y=90)

        tk.Label(self.inner_canvas, text="HTML file name",
             font=("Helvetica", 14)).place(x=80, y=125)

        tk.Label(self.inner_canvas, text="Plot Type",
             font=("Helvetica", 14)).place(x=80, y=160)
        
        self.e1 = tk.Entry(self.inner_canvas)
        self.e2 = tk.Entry(self.inner_canvas)
        self.e3 = tk.Entry(self.inner_canvas)
        self.e4 = tk.Entry(self.inner_canvas)
        self.e5 = tk.Entry(self.inner_canvas)

        self.wordcanvas.create_window(280,30 , window=self.e1)
        self.wordcanvas.create_window(280,70 , window=self.e2)
        self.wordcanvas.create_window(280,105 , window=self.e3)
        self.wordcanvas.create_window(280,140 , window=self.e4)
        self.wordcanvas.create_window(280,180 , window=self.e5)
        

    def run_app(self):

        val1 = self.e1.get()
        val2 = self.e2.get()
        val3 = self.e3.get()
        val4 = self.e4.get()
        val5 = self.e5.get()

        try:
            
            word = GetNameEntity(val1)
            word.namentity(val2,val3,val4,val5)

        except Exception as e:

            messagebox.showerror("showerror",e)

    def buttons_inner(self):

        tk.Button(self.inner_canvas, text='Create Html', command=self.run_app).place(x=120,y=200)

        self.wordcanvas.pack()
        self.inner_canvas.mainloop()

class GUI:

    def __init__(self):

        self.top = tk.Tk()
        self.top.geometry('400x250')
        self.top.eval('tk::PlaceWindow . center')
        self.canvas = tk.Canvas(self.top, width=600, height=400)

        tk.Label(self.top, text="Select CSV file",
             font=("Helvetica", 14)).place(x=80, y=30)
        
        tk.Label(self.top, text="Feature Name",
             font=("Helvetica", 14)).place(x=80, y=65)
        
        tk.Label(self.top, text="HTML file name",
             font=("Helvetica", 14)).place(x=80, y=100)

        tk.Label(self.top, text="Plot Type",
             font=("Helvetica", 14)).place(x=80, y=140)
        
        self.e1 = tk.Entry(self.top)
        self.e2 = tk.Entry(self.top)
        self.e3 = tk.Entry(self.top)
        self.e4 = tk.Entry(self.top)

        self.canvas.create_window(280,40 , window=self.e1)
        self.canvas.create_window(280,75 , window=self.e2)
        self.canvas.create_window(280,115 , window=self.e3)
        self.canvas.create_window(280,150 , window=self.e4)

    def run_app(self):

        val1 = self.e1.get()
        val2 = self.e2.get()
        val3 = self.e3.get()
        val4 = self.e4.get()

        try:
            name = NameEntityRecognition(val1)
            name.namentity(val2,val3,val4)
        except Exception as e:

            messagebox.showerror("showerror",e)

    
    def filter_by_word(self):

        self.word = WordGUI()
        self.word.buttons_inner()

    def buttons(self):

        tk.Button(self.top, text='Create HTML', command=self.run_app).place(x=120,y=180)
        tk.Button(self.top, text='filter a word', command=self.filter_by_word).place(x=220,y=180)

        self.canvas.pack()
        self.top.mainloop()
    
    
if __name__ == '__main__':

    gui = GUI()
    gui.buttons()