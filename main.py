import nltk
from flask import Flask
import socket
import pandas
from IPython.display import display

# from db import db_import_data_testing, db_import_data_training, get_hasil, get_label

# import preprocess
from process import jarakeuclideanDTDS, ranking, tfidf, jarakeuclideanDTDT, small, k11_euclidean, lokasi, pelabelan, validitas

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

    # x=tfidf(get_hasil())
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
    df = pandas.read_csv('D:/github/mknn/Upload/tfidf.csv', sep = ',')
    # display(df)
    # split1 = (df.shape[0]*8)/10
    split1 = (150*8)/10
    # split2 = (len(df)*2)/100
    
    # display(df.iloc[121:149])
    df1 = pandas.DataFrame(df.iloc[:int(split1)])
    df2 = pandas.DataFrame(df.iloc[int(split1):150])
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)
    # df = pandas.read_csv('D:/github/mknn/Upload/euclideandtdt.csv', sep = ',')
    
    jarak_dtdt = pandas.read_csv('D:/github/mknn/Upload/euclideandtdt.csv', sep = ',')
    jarak_dtds = pandas.read_csv('D:/github/mknn/Upload/euclideandtds.csv', sep = ',')

    small_dtdt = small(jarak_dtdt)
    small_dtds = small(jarak_dtds)
    small_dtdt.to_csv('D:/github/mknn/Upload/smalleuclideandtdt.csv', index=False)
    small_dtds.to_csv('D:/github/mknn/Upload/smalleuclideandtds.csv', index=False)
    # df = pandas.read_csv('D:/github/mknn/Upload/smalleuclideandtdt.csv', sep = ',')
    display(df)
    k11_dtdt = k11_euclidean(small_dtdt)
    k11_dtds = k11_euclidean(small_dtds)

    # df = pandas.read_csv('D:/github/mknn/Upload/euclideandtdt.csv', sep = ',')
    display(df)
    lokasi_dtdt = lokasi(jarak_dtdt)
    lokasi_dtds = lokasi(jarak_dtds)
    lokasi_dtdt.to_csv('D:/github/mknn/Upload/lokasiargsmalleuclideandtdt.csv', index=False)
    lokasi_dtds.to_csv('D:/github/mknn/Upload/lokasiargsmalleuclideandtds.csv', index=False)
    # df1 = pandas.read_csv('D:/github/mknn/Upload/argsmalleuclideandtdt.csv', sep = ',')
    df2 = pandas.read_csv('D:/github/mknn/Upload/traininglabel.csv', sep = ',')
    display(df1)
    display(df2)
    pelabelan_dtdt = pelabelan(lokasi_dtdt, df2)
    pelabelan_dtds = pelabelan(lokasi_dtds, df2)
    pelabelan_dtdt.to_csv('D:/github/mknn/Upload/labelargsmalleuclideandtdt.csv', index=False)
    pelabelan_dtds.to_csv('D:/github/mknn/Upload/labelargsmalleuclideandtds.csv', index=False)
    # df1 = pandas.read_csv('D:/github/mknn/Upload/labelargsmalleuclideandtdt.csv', sep = ',')
    df2 = pandas.read_csv('D:/github/mknn/Upload/traininglabel.csv', sep = ',')
    display(df1)
    display(df2)
    validitas_dtdt = validitas(pelabelan_dtdt, df2)
    validitas_dtds = validitas(pelabelan_dtds, df2)
    validitas_dtdt.to_csv('D:/github/mknn/Upload/valargsmalleuclideandtdt.csv', index=False)
    validitas_dtds.to_csv('D:/github/mknn/Upload/valargsmalleuclideandtds.csv', index=False)

    ranking, label = ranking(validitas_dtdt, validitas_dtds)
    ranking.to_csv('D:/github/mknn/Upload/ranking.csv', index=False)
    print(label)
    lbl = []
    for i in label:
        lbl.append(df2['0'][i])
    print(max(set(lbl), key = lbl.count))
