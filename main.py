import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import nltk
from wordcloud import WordCloud

os.chdir(r"C:\Users\Swag\Documents\PersonalProject")


#data from kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text
text = pd.read_csv("AIgeneratedtext.csv")

###Basic analysis using tf idf and SVM###

def tokenize(textData):
    t = []
    for i in range(len(textData)):
        t.append(nltk.WhitespaceTokenizer().tokenize(text['Text'].iloc[i]))
    textData['Tokenized'] = t
    return

tokenize(text)

def remPunctuation(textData, tokenized):
    t = []
    for i in range(len(textData)):
        for j in range(len(textData[tokenized].iloc[i])):
            for k in '!?.()\/:;"-':
                if k in textData[tokenized].iloc[i][j]:
                    text = ''
                    for l in range(len(textData[tokenized].iloc[i][j])):
                        if textData[tokenized].iloc[i][j][l]!=k:
                            text = text + textData[tokenized].iloc[i][j][l]
                    textData[tokenized].iloc[i][j] = text
    return

remPunctuation(text, 'Tokenized')

def tfidf(textData, textLabel):
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer()
    tfidf = tfidf.fit_transform(textData[textLabel])
    tfidf = tfidf.toarray()

    t = []
    for i in range(tfidf.shape[0]):
        t.append(list(tfidf[i]))
    textData['tf idf'] = t
    return

tfidf(text,'Text')


def tfidf_array(textData, textLabel):
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer()
    tfidf = tfidf.fit_transform(textData[textLabel])
    tfidf = tfidf.toarray()
    return tfidf

tf_idf = tfidf_array(text,'Text')

def numlabel(textData, textLabel):
    t = []
    for i in range(len(textData)):
        if textData[textLabel].iloc[i] == "student":
            tt = 0
        else:
            tt = 1

        t.append(tt)
    textData['NumLabel'] = t
    return

numlabel(text,'Label')

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(tf_idf, np.array(text['NumLabel']),test_size = 0.25)

model = sklearn.svm.LinearSVC()
model.fit(X_train,Y_train)

Y_predicted = model.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(Y_test,Y_predicted)
print(accuracy)

###Other information that can be gleamed from the data###

###Most common words###

def mostCommonWords(textData, textTokenized):
    mcw = {}
    for i in range(len(textData)):
        for j in textData[textTokenized].iloc[i]:
            if j in mcw:
                mcw[j] = mcw[j]+1
            else:
                mcw[j] = 1
    return mcw

mcwAI = mostCommonWords(text[text['Label']=='ai'], 'Tokenized')
mcwStudent = mostCommonWords(text[text['Label']=='student'], 'Tokenized')

mcwAI_df = pd.DataFrame.from_dict(mcwAI, orient='index', columns = ['Freq'])
mcwAI_df['Word'] = list(mcwAI)
mcwAI_df = mcwAI_df.sort_values(by='Freq',ascending=False)

mcwStudent_df = pd.DataFrame.from_dict(mcwStudent, orient='index', columns = ['Freq'])
mcwStudent_df['Word'] = list(mcwStudent)
mcwStudent_df = mcwStudent_df.sort_values(by='Freq',ascending=False)

wc = WordCloud().generate_from_frequencies(mcwAI)
plt.title('Most Common Words in AI Generated Text')
plt.imshow(wc)
plt.show()

wc = WordCloud().generate_from_frequencies(mcwStudent)
plt.title('Most Common Words in Student/Human Generated Text')
plt.imshow(wc)
plt.show()

plt.bar(range(0,50),mcwAI_df['Freq'].iloc[0:50])
plt.xticks(range(0,50),mcwAI_df['Word'].iloc[0:50],rotation='vertical')
plt.title('Most Common Words in AI Generated Text')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.show()

plt.bar(range(0,50),mcwStudent_df['Freq'].iloc[0:50])
plt.xticks(range(0,50),mcwStudent_df['Word'].iloc[0:50],rotation='vertical')
plt.title('Most Common Words in Student/Human Generated Text')
plt.show()
