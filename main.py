from flask import Flask
import socket

from db import db_import_data_testing, db_import_data_training

import preprocess
import process

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
    data = preprocess.get_data_preprocessing("Upload/training.csv")
    data_testing = preprocess.get_data_preprocessing("Upload/testing.csv")

    data_ready = []
    for list_data in data:
        data_ready.append(list_data[5].split(" "))

    test = process.get_tf_idf(data_testing[0][5].split(" "), data_ready)
    # preprocess.preprocessing("@ctigeek Hi Steven, this was improved in recent Windows Insider builds. Here's more information regarding the improvement: https://t.co/s4A4ayqPyZ")
