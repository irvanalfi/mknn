from flask import Flask
import socket

from db import db_import_data_training, db_get_all_training

import preprocess

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)

@app.route('/' , methods=['GET','POST'])
def dashboar():
    data = {
        "datatesting": "200",
        "datatraining": "800"
    }
    return data

if __name__ == "__main__":
    # app.run(host=my_ip)
    # db_import_data_training()
    # db_get_all_training()
    db_import_data_training("Upload/training.csv")
    # preprocess.preprocessing("@ctigeek Hi Steven, this was improved in recent Windows Insider builds. Here's more information regarding the improvement: https://t.co/s4A4ayqPyZ")