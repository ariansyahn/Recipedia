import PyPDF2
import docx2txt
from os import listdir
from os.path import isfile, join
import math
import string
import re
import os

from collections import OrderedDict, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce
from operator import itemgetter
from flask import Flask, render_template, request
from flask_wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired, Email, Length, AnyOf

app = Flask(__name__)

hasilDict = {"test": ""}
# allContentDict = {"content": ""}


class Document():
    # constructor
    def __init__(self, id, name, text, path):
        self.id = id
        self.name = name
        # self.text = text + ' '+name
        self.text = text
        self.path = path


def get_content_file(directory):
    # extensions = ['pdf', 'docx', 'txt']
    document_contents = []
    id_doc = 0

    for filename in os.listdir(directory):
        path = directory+'\\'+filename
        text = ""
        if(filename[len(filename)-3:] == 'pdf'):
            pdf_file = open(path, 'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            num_pages = pdf_reader.numPages
            for i in range(num_pages):
                page = pdf_reader.getPage(i)
                text += page.extractText()
        elif(filename[len(filename)-4:] == 'docx'):
            text = docx2txt.process(path)
        elif(filename[len(filename)-3:] == 'txt'):
            f = open(path, 'r')
            for t in f.readlines():
                text += ' '+t

        document = Document(id_doc, filename, text, path)
        document_contents.append(document)
        id_doc += 1
    return document_contents


all_document = get_content_file('static/data')
N = len(all_document)
dictionary = set()
# ini index
postings = defaultdict(dict)
document_frequency = defaultdict(int)
length = defaultdict(float)
termInDocument = []


def preprocessing(document):

    # removing number
    document = re.sub('[0-9]+', '', document)
    # case folding
    document = document.lower()
    # remove punctuation
    document = re.sub(r'^https?:\/\/.*[\r\n]*',
                      '', document, flags=re.MULTILINE)
    tokens = document.split()
    punc = string.punctuation
    tokens = [token.strip(punc) for token in tokens]
    # stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    stemmer = PorterStemmer()
    for i in range(0, len(tokens)):
        if (tokens[i] != stemmer.stem(tokens[i])):
            tokens[i] = stemmer.stem(tokens[i])
    # print(tokens)  # check
    # allContentDict["content"] = tokens
    return tokens


def set_terms_and_postings():
    global dictionary, postings, termInDocument
    # termInDocument2 = []
    for doc in all_document:
        terms = preprocessing(doc.text)
        unique_terms = set(terms)
        unique_terms = sorted(unique_terms)
        termInDocument.append(list(unique_terms))
        dictionary = dictionary.union(unique_terms)
        for term in unique_terms:
            postings[term][doc.id] = terms.count(term)
            # print(postings[term][doc.id])


set_terms_and_postings()
# print(termInDocument)
# print(type(dictionary))
# print(len(dictionary))
output_dict = {item: val for val, item in enumerate(dictionary)}
output_dict = sorted(output_dict)
print("Index Created with Total : ", len(postings))


def set_document_frequencies():
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])


set_document_frequencies()
# print(document_frequency)


def set_inverse_document_frequency(term):
    if term in dictionary:
        return math.log10(float(N)/float(document_frequency[term]))
    else:
        return 0.0


def imp(term, id):
    if id in postings[term]:
        return postings[term][id]*set_inverse_document_frequency(term)
    else:
        return 0.0


def set_lengths():
    global length
    for doc in all_document:
        l = 0
        for term in dictionary:
            l += imp(term, doc.id)**2
        length[doc.id] = math.sqrt(l)


set_lengths()


def intersection(sets):
    return reduce(set.intersection, [s for s in sets])


def similarity(query, id):
    similarity = 0.0
    for term in query:
        if term in dictionary:
            similarity += set_inverse_document_frequency(term)*imp(term, id)
    similarity = similarity/length[id]

    return similarity


def find_doc_by_id(id):
    for d in all_document:
        if(d.id == id):
            return (d.name, d.path, d.id)


class ResultDocument():
    def __init__(self, score, name, path, id):
        self.score = score
        self.name = name
        self.path = path
        self.id = id


def search(query):
    query = preprocessing(query)
    all_result_document = []
    id_set = []
    relevant_document_ids = intersection(
        [set(postings[term].keys()) for term in query])
    # print(relevant_document_ids)
    for term in query:
        for id_d in postings[term].keys():
            id_set.append(id_d)
    id_set = set(id_set)

    # if not relevant_document_ids:
    if not id_set:
        print("No documents matched all query terms.")
    else:
        scores = sorted([(id, similarity(query, id))
                         for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
        for (id, score) in scores:
            detail = find_doc_by_id(id)
            # dari return find_doc, dalam bentuk array
            re_doc = ResultDocument(score, detail[0], detail[1], detail[2])
            all_result_document.append(re_doc)
    return top_ten(all_result_document)


# print(search("pork"))


def top_ten(results):
    if len(results) > 10:
        return results[:10]
    else:
        return results


def vector(text, terms):
    Vec = []
    for i in range(len(terms)):
        if(terms[i] in text):
            Vec.append(1)
        else:
            Vec.append(0)
    return Vec


def relevance(listDoc, allContent):
    relevance = []
    for i in range(len(listDoc)):
        if(i < 5):
            relevance.append(allContent[int(listDoc[i])])

    # relevances = preprocessing(relevance)
    return relevance


def sumVector(VectorList):
    results = []
    results.append(VectorList[0])

    for i in range(1, len(VectorList)):
        for j in range(len(VectorList[i])):
            results[0][j] = results[0][j] + VectorList[i][j]

    return results[0]


def multiplyVector(alpha, listVector):
    res = []
    for i in range(len(listVector)):
        res.append(alpha*int(listVector[i]))
    return res


# ROUTING


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def searching():
    if request.method == 'POST':
        cari = request.form['search_input']
        hasil = []
        array = []
        try:
            for d in search(cari):
                re = {"score": d.score,
                      "path": d.path,
                      "name": d.name,
                      "id": d.id
                      }
                # print(d.id)
                hasil.append(re)
                # hasilDict["test"]
                array.append(d.id)
            if hasil:
                hasilDict["test"] = array
                # print("ahahahaha" + hasilDict["test"])
                # hasilDict["test"] = hasil[id]
                return render_template('index.html', result=hasil, query=cari, array=hasilDict["test"])
            else:
                render_template('index.html')
        except Exception:
            pass

    return render_template('index.html')


# gg = find_doc_by_id(1)


@app.route('/indexer')
def indexPage():
    return render_template('indexer.html', index=postings)


# print(postings)

# print(hasilDict["test"])


@app.route('/expand/<string:query>')
def expand(query):
    queryVec = vector(query, list(output_dict))
    relVec = []
    total = []
    totalDict = {}
    # print(queryVec)
    res = relevance(hasilDict["test"], termInDocument)
    # print(res)
    for words in res:
        relVec.append(vector(words, list(output_dict)))
    # print(relVec)
    relVecSum = sumVector(relVec)
    # alpha=1
    mulQuery = multiplyVector(1, queryVec)
    # beta=0.8
    mulRelevan = multiplyVector(0.8, relVecSum)

    # jumlah
    for i in range(len(mulQuery)):
        total.append(mulQuery[i] + mulRelevan[i])

    # buat dict
    for i in range(len(mulQuery)):
        if(total[i] > 0):
            totalDict[output_dict[i]] = total[i]

    new = OrderedDict(
        sorted(totalDict.items(), key=itemgetter(1), reverse=True))
    # print(new)
    new_query = {}
    for i in range(6):
        new_query[str(list(new.keys())[i])] = list(new.values())[i]
    print(new_query)
    str_query = []
    for i in range(len(new_query)):
        str_query.append(list(new_query.keys())[i])
    str_query = " ".join(str_query)
    print(str_query)

    # search again
    hasil = []
    array = []
    try:
        for d in search(str_query):
            re = {"score": d.score,
                  "path": d.path,
                  "name": d.name,
                  "id": d.id
                  }
            # print(d.id)
            hasil.append(re)
            # hasilDict["test"]
            array.append(d.id)
        if hasil:
            hasilDict["test"] = array
            # print("ahahahaha" + hasilDict["test"])
            # hasilDict["test"] = hasil[id]
            return render_template('index.html', result=hasil, query=str_query, array=hasilDict["test"])
        else:
            render_template('index.html')
    except Exception:
        pass
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
