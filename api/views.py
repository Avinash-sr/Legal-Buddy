from django.shortcuts import render

from rest_framework.response import Response
from .models import *
from rest_framework.decorators import api_view
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from .models import *


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

from .serializers import *


custom_objects = {'KerasLayer': hub.KerasLayer}
model = tf.keras.models.load_model("E:\ML\model_2(72% ts).h5", custom_objects=custom_objects)

CLASSES=['POCSO-2012', 'IPC-509', 'IPC-504', 'IPC-354', 'IPC-354A', 'IPC-498A',
       'IPC-302', 'IPC-376', 'DPA', 'DVA', 'IPC-34', 'IPC 307', 'IPC 313',
       'IPC 109', 'IPC 324', 'IPC 326', 'IPC 323']
#--------------------------------------------------------------------

@api_view(['POST'])
def getresult(request):
    module_path = "E:/ML/universal_sentence_encoder_1"
    if request.method == 'POST':
        #1 preproceessing
        sentences = request.data
        sentences = [sentences]
        sentences = PreprocessingLayer(sentences)

        #2 predict

        # predict function call
        result = predict(sentences, model, CLASSES)

        predict = []
        # Assuming laws is your model and lawsSerializer is the serializer for it
        for code in result:
            m = laws.objects.get(code=code)
            serialized_data = lawsSerailizer(m).data
            predict.append(serialized_data)
        
        return Response(predict)
        
def predict(case, model, classes):
    print(case)
    pred = model.predict(case)[0]
    top5_indices = np.argsort(pred)[::-1][:5]
    top5_indices = top5_indices.astype(int)  # Ensure top5_indices is of type int
    top5_classes = [classes[i] for i in top5_indices]
    top5_probabilities = pred[top5_indices]
    pred= []
    for i in range(5):
        print(f"{i + 1}. Class: {top5_classes[i]}, Probability: {top5_probabilities[i]:.4f}") 
        pred.append(top5_classes[i])
    return pred


# user helper function
def lowercase(sentences):
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
    return sentences

def remove_punctuations(sentences):
    cleaned_sentences= []
    for sent in sentences :
        words = word_tokenize(sent)
        word_without_punt = [word for word in words if word not in string.punctuation]
        sentence_without_punct = ' '.join(word_without_punt)
        cleaned_sentences.append(sentence_without_punct)

    return cleaned_sentences

def remove_stopwords(sentences):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words_without_stop = [word for word in words if word.lower() not in stop_words]
        sent_without_stop = ' '.join(words_without_stop)
        filtered_sentences.append(sent_without_stop)

    return filtered_sentences

def PreprocessingLayer(sentences):
    sentences = lowercase(sentences)
    sentences = remove_punctuations(sentences)
    sentences = remove_stopwords(sentences)

    return sentences

 