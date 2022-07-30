from flask import Flask
import socket
import pandas
from tqdm import tqdm
from IPython.display import display

from db import db_import_data_testing, db_import_data_training, get_hasil, get_label, updateData
from preprocess import *
from process import *

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def dashboar():
    data = {
        "datatesting": "200",
        "datatraining": "800"
    }
    return data


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


if __name__ == "__main__":
    # app.run(host=my_ip)
    # db_get_all_training()
    # db_import_data_training("Upload/training.csv")
    # db_import_data_testing("Upload/testing.csv")
    # pelabelanOtomatis()
    # x= tfidf(get_hasil())

    df = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')

    split1 = (df.shape[0] * 6) / 10

    df1 = pandas.DataFrame(df.iloc[:int(split1)])
    df2 = pandas.DataFrame(df.iloc[int(split1):])
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)

    jarak_dtdt = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=',')
    jarak_dtds = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=',')

    small_dtdt = small(jarak_dtdt)

    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    idx = 1
    for j in tqdm(k):
        idx = 1
        nama_polaritas = "polaritas_akhir_k" + str(j)
        k_dtdt = k_euclidean(small_dtdt, j)
        k_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/k11euclideandtdt.csv', index=False)

        lokasi_dtdt = lokasi(jarak_dtdt, j)
        lokasi_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/lokasiargsmalleuclideandtdt.csv', index=False)

        label_train = get_label("training")
        label_test = get_polaritas("polaritas_awal")

        pelabelan_dtdt = pelabelan(lokasi_dtdt, label_train)
        pelabelan_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelargsmalleuclideandtdt.csv', index=False)

        validitas_dtdt = validitas(pelabelan_dtdt, label_train)
        validitas_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/valargsmalleuclideandtdt.csv', index=False)

        lbl = []
        for i in tqdm(range(len(jarak_dtds.iloc[0])), leave=False):
            rank, label = ranking(validitas_dtdt, jarak_dtds[str(i)], j)
            l = pandas.DataFrame(label)
            lbl2 = []
            for i in label:
                lbl2.append(label_train[i])
            lbl.append(most_frequent(lbl2))
            updateData(nama_polaritas, most_frequent(lbl2), idx)
            idx += 1
        label_df = pandas.DataFrame(lbl)

    for j in k:
        print("pengujian akurasi k-" + str(j))
        testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(j))
