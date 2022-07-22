import nltk
from flask import Flask
import socket
import pandas
from IPython.display import display
from db import db_import_data_testing, db_import_data_training, get_hasil, get_label

import preprocess
from process import tfidf, jarakeuclideanDTDT, small

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def dashboar():
    data = {
        "datatesting": "200",
        "datatraining": "800"
    }
    return data


if __name__ == "__main__":
    # app.run(host=my_ip)
    # db_get_all_training()
    # db_import_data_training("Upload/training.csv")
    # db_import_data_testing("Upload/testing.csv")

    x=tfidf(get_hasil())
    #
    # print(x)
    # get_label(data='training')
    # bow_vectorize = TfidfVectorizer()
    # res = bow_vectorize.fit_transform(get_hasil())
    # print(res)



    # hasil = get_hasil(data="training")
    # df= pd.DataFrame(bow_vectorize.todense().T, index=text_bow, columns=[f'D{i+1}' for i in range(len(hasil))])
    # df

    # tfidf()
    # db_import_data_testing("Upload/testing.csv")
    # data = preprocess.get_data_preprocessing("Upload/training.csv")
    # data_testing = preprocess.get_data_preprocessing("Upload/testing.csv")
    #
    # data_ready = []
    # for list_data in data:
    #     data_ready.append(list_data[5].split(" "))
    #
    # test = process.get_tf_idf(data_testing[0][5].split(" "), data_ready)
    # preprocess.preprocessing("@ctigeek Hi Steven, this was improved in recent Windows Insider builds. Here's more information regarding the improvement: https://t.co/s4A4ayqPyZ")
    # df = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidftraining.csv', sep = ',')
    # display(df)
    # jarakeuclideanDTDT(df)
    # df = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep = ',')
    # display(df)
    # small(df)