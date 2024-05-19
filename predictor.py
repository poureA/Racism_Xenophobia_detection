# Import necessary modules.

from tkinter import Tk , Label , Button , filedialog
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Predictor:
    def __doc__(self)->str:
        return '''Class Docstring'''
    
    def Filter(self,text)->str:
        '''Function Docstring'''
        
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
        result = [word for word in text.lower().split() if word not in stopwords]
        return ' '.join(result)
    
    def predict(self)->None:
        '''Function Docstring'''
        
        #Creating the main window and setting the background color of the window.
        
        root = Tk()
        root.config(background='light blue')
        
        #Label to display the content of the selected text file.
        
        board = Label(text='Your text file content will appear here',font=(15)).grid(row=0,column=0,ipadx=100,ipady=70)
        
        #Function to open a file dialog and read the content of the selected text file.
        
        def File_explorer():
            filename = filedialog.askopenfilename(filetypes=(('Text files', '*.txt*'),
                                                             ('All files', '*.*')))
            with open(filename,'r') as file:
                global content
                content = file.read()
                
                #Update the label with the content of the text file.
                
                board = Label(text=content,font=(15)).grid(row=0,column=0,ipadx=100,ipady=70)
                return board
            
        #Button to open the file explorer. 
        
        File_btn = Button(text='Open a text file',bg='yellow',command=File_explorer).grid(ipadx=5,ipady=3)
        
        #Function to predict racism and xenophobia in the text content.
        
        def Do():
            
            # Load the racism detection model and tokenizer
            
            model_r = load_model('Racism_detection_model')
            with open('tknizer_R.pkl','rb') as file:
                tokenizer_r = load(file)
                
            #Preprocess the text content for racism detection.
                
            sequence_r = tokenizer_r.texts_to_sequences([self.Filter(content)])
            matrix_r = pad_sequences(sequence_r,padding='post',truncating='post',maxlen=16)
            pred_r = model_r.predict(matrix_r)
            
            #Load the xenophobia detection model and tokenizer.
            
            model_x = load_model('Xenophobia_detection_model')
            with open('tknizer_X.pkl','rb') as file_xeno:
                tokenizer_x = load(file_xeno)
                
            #Preprocess the text content for xenophobia detection.
            
            sequence_x = tokenizer_x.texts_to_sequences([self.Filter(content)])
            matrix_x = pad_sequences(sequence_x,padding='post',truncating='post',maxlen=50)
            pred_x = model_x.predict(matrix_x)
            
            # Display the predictions.
            
            result = Label(text=f'Racism: {pred_r[0][0]*100:0.2f}%\nXenophobic: {pred_x[0][0]*100:0.2f}%',bg='light gray',font=(10)).grid(row=4,column=0,ipadx=10,ipady=15)
            return result 
        
        #Button to evaluate the content for racism and xenophobia.
        
        predict_btn = Button(text='Evaluate',bg='light green',command=Do).grid(ipadx=5,ipady=3)
        
        #Label to display the initial prediction results.
        
        result = Label(text='Rasism:\nXenophobia:',bg='light gray',font=(10)).grid(row=4,column=0,ipadx=10,ipady=15)
        
        
        # Set the title of the main window and Run the Tkinter event loop

        root.title('Predictor')
        root.mainloop()

#Create an instance of the Predictor class and call the predict method.

sample = Predictor()
sample.predict()
