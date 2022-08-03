import pandas
from tqdm import tqdm
from db import db_import_data_testing, db_import_data_training, get_hasil, get_label, updateData
from preprocess import *
from process import *


if __name__ == "__main__":
    # db_import_data_training("Upload/training.csv")
    # db_import_data_testing("Upload/testing.csv")
    # # proses tf-idf training dan testing
    # x= tfidf(get_hasil())
    # # membaca csv hasil tf-idf
    df = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')
    # membagi hasil tf idf dari total data set ke 600 training, 400 testing
    split1 = (df.shape[0] * 6) / 10
    # baca hasil split tf-idf data training dan testing
    df1 = pandas.DataFrame(df.iloc[:int(split1)])
    df2 = pandas.DataFrame(df.iloc[int(split1):])
    # proses jarak euclidean dan menyimpan data ke dalam file csv
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)
    # read csv jarak_dtdt dan jarak_dtds
    jarak_dtdt = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=',')
    jarak_dtds = pandas.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=',')
    # mengurutkan data jarak_dtdt
    small_dtdt = small(jarak_dtdt)
    #jumlah k yang digunakan
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
            labelterdekat = pandas.DataFrame(lbl2)
            labelterdekat.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelterdekat.csv')
            lbl.append(most_frequent(lbl2))
            updateData(nama_polaritas, most_frequent(lbl2), idx)
            idx += 1
        label_df = pandas.DataFrame(lbl)

    # k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # for j in k:
    #     print("pengujian akurasi k-" + str(j))
    #     testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(j))
