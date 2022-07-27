from flask import Flask
import socket
import pandas
from tqdm import tqdm
from IPython.display import display

from db import db_import_data_testing, db_import_data_training, get_hasil, get_label, updateData
# import preprocess
from process import jarakeuclideanDTDS, ranking, tfidf, jarakeuclideanDTDT, small, k11_euclidean, lokasi, pelabelan, validitas, testaccuracy

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

    # x=tfidf(get_hasil())
    # get_label(data='training')
    # bow_vectorize = TfidfVectorizer()
    # res = bow_vectorize.fit_transform(get_hasil())
    # hasil = get_hasil(data="training")
    # df= pd.DataFrame(bow_vectorize.todense().T, index=text_bow, columns=[f'D{i+1}' for i in range(len(hasil))])
    # df

    # data = preprocess.get_data_preprocessing("Upload/training.csv")
    # data_testing = preprocess.get_data_preprocessing("Upload/testing.csv")
    #
    # data_ready = []
    # for list_data in data:
    #     data_ready.append(list_data[5].split(" "))
    #
    # test = process.get_tf_idf(data_testing[0][5].split(" "), data_ready)

    df = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')
    display(df)
    # split1 = (df.shape[0]*8)/10
    split1 = (df.shape[0] * 9) / 10
    # split2 = (len(df)*2)/100

    # display(df.iloc[121:149])
    df1 = pandas.DataFrame(df.iloc[:int(split1)])
    df2 = pandas.DataFrame(df.iloc[int(split1):])
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)
    # df = pandas.read_csv('D:/github/mknn/Upload/euclideandtdt.csv', sep = ',')

    jarak_dtdt = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=',')
    jarak_dtds = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=',')

    small_dtdt = small(jarak_dtdt)
    # small_dts = small(jarak_dtds)
    small_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/smalleuclideandtdt.csv', index=False)
    # small_dtds.to_csv('D:/github/mknn/Upload/smalleuclideandtds.csv', index=False)
    # df = pandas.read_csv('D:/github/mknn/Upload/smalleuclideandtdt.csv', sep = ',')
    display(df)
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
    idx = 1
    for j in tqdm(k):
        idx = 1
        nama_polaritas = "polaritas_akhir_k" + str(j)
        k11_dtdt = k11_euclidean(small_dtdt, j)
        k11_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/k11euclideandtdt.csv', index=False)
        # k11_dtds = k11_euclidean(small_dtds)

        # df = pandas.read_csv('D:/github/mknn/Upload/euclideandtdt.csv', sep = ',')
        display(df)
        lokasi_dtdt = lokasi(jarak_dtdt, j)
        # lokasi_dtds = lokasi(jarak_dtds)
        lokasi_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/lokasiargsmalleuclideandtdt.csv', index=False)
        # lokasi_dtds.to_csv('D:/github/mknn/Upload/lokasiargsmalleuclideandtds.csv', index=False)
        # df1 = pandas.read_csv('D:/github/mknn/Upload/argsmalleuclideandtdt.csv', sep = ',')
        df2 = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/training.csv', sep=';', header=None)[2]
        df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/label_train.csv', index=False)
        df2 = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/testing.csv', sep=';', header=None)[2]
        df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/label_test.csv', index=False)
        display(df1)
        display(df2)
        df2 = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/label_test.csv', sep=';')
        pelabelan_dtdt = pelabelan(lokasi_dtdt, df2)
        # pelabelan_dtds = pelabelan(lokasi_dtds, df2)
        pelabelan_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelargsmalleuclideandtdt.csv', index=False)
        # pelabelan_dtds.to_csv('D:/github/mknn/Upload/labelargsmalleuclideandtds.csv', index=False)
        # df1 = pandas.read_csv('D:/github/mknn/Upload/labelargsmalleuclideandtdt.csv', sep = ',')
        df2 = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/label_test.csv', sep=';')
        display(df1)
        display(df2)
        validitas_dtdt = validitas(pelabelan_dtdt, df2)
        # validitas_dtds = validitas(pelabelan_dtds, df2)
        validitas_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/valargsmalleuclideandtdt.csv', index=False)
        # validitas_dtds.to_csv('D:/github/mknn/Upload/valargsmalleuclideandtds.csv', index=False)
        lbl = []
        for i in tqdm(range(len(jarak_dtds.iloc[0])), leave=False):
            rank, label = ranking(validitas_dtdt, jarak_dtds[str(i)], j)
            l = pandas.DataFrame(label)
            l.to_csv('C:/Users/IRVAN/backendmknn/Upload/label.csv', index=False)
            # print(label)
            lbl2 = []
            for i in label:
                lbl2.append(df2['2'][i])
                # print(i)
            # lbl2.sort()
            lbl.append(most_frequent(lbl2))
            updateData(nama_polaritas, most_frequent(lbl2), idx)
            idx += 1
            # print(lbl2.count('positif'), lbl2.count('netral'), lbl2.count('negatif'))
        # print(lbl2)
        label_df = pandas.DataFrame(lbl)

    # k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # for j in k:
    #     print("pengujian akurasi k-" + str(j))
    #     testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(j))