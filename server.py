from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tqdm import tqdm
import socket
from db import *
from process import *
from crawling import *
import pandas as pd

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return "server berjalan"


@app.route('/crawling', methods=['GET', 'POST'])
def proses_crawling():
    keyword = request.form.get("keyword")
    data = crawling(keyword)
    if bool(data):
        response = {
            'status': 'success'
        }
    else:
        response = {
            'status': 'failed'
        }
    return response


@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    data = get_user_by_uname_pass(username, password)
    if bool(data):
        response = {
            'status': 'success',
            'data': data
        }
    else:
        response = {
            'status': 'failed',
            'data': data
        }
    return response


@app.route('/dashboard-data', methods=['GET', 'POST'])
def dashboard_data():
    training_positif = count_polaritas("positif", "training", "polaritas")
    training_negatif = count_polaritas("negatif", "training", "polaritas")
    training_netral = count_polaritas("netral", "training", "polaritas")
    total_training = training_positif + training_negatif + training_netral
    testing_positif = count_polaritas("positif", "testing", "polaritas_awal")
    testing_negatif = count_polaritas("negatif", "testing", "polaritas_awal")
    testing_netral = count_polaritas("netral", "testing", "polaritas_awal")
    total_testing = testing_positif + testing_negatif + testing_netral
    dataset_postif = training_positif + testing_positif
    dataset_negatif = training_negatif + testing_negatif
    dataset_netral = training_netral + testing_netral
    total_dataset = dataset_postif + dataset_negatif + dataset_netral
    k = ['K3', 'K5', 'K7', 'K9', 'K11', 'K13', 'K15', 'K17', 'K19', 'K21']
    data = {
        "jml_training_positif": training_positif,
        "jml_training_negatif": training_negatif,
        "jml_training_netral": training_netral,
        "total_training": total_training,
        "jml_testing_positif": testing_positif,
        "jml_testing_negatif": testing_negatif,
        "jml_testing_netral": testing_netral,
        "total_testing": total_testing,
        "jml_dataset_positif": dataset_postif,
        "jml_dataset_negatif": dataset_negatif,
        "jml_dataset_netral": dataset_netral,
        "total_dataset": total_dataset,
        "k": k,
        "jml_k": len(k)
    }
    return jsonify(data)


@app.route('/dashboard-chart', methods=['GET', 'POST'])
def dashboard_chart():
    polaritas_netral = []
    polaritas_positif = []
    polaritas_negatif = []
    sentimen = ['netral', 'positif', 'negatif']
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for i in k:
        temp_netral = db_get_count_polaritas('k' + str(i), sentimen[0])
        polaritas_netral.append(temp_netral)
    for i in k:
        temp_positif = db_get_count_polaritas('k' + str(i), sentimen[1])
        polaritas_positif.append(temp_positif)
    for i in k:
        temp_negatif = db_get_count_polaritas('k' + str(i), sentimen[2])
        polaritas_negatif.append(temp_negatif)
    polaritas_netral = ",".join([str(i) for i in polaritas_netral])
    polaritas_positif = ",".join([str(i) for i in polaritas_positif])
    polaritas_negatif = ",".join([str(i) for i in polaritas_negatif])
    data = {
        'polaritas_positif': polaritas_positif,
        'polaritas_negatif': polaritas_negatif,
        'polaritas_netral': polaritas_netral
    }
    return jsonify(data)


@app.route('/training-data', methods=['GET', 'POST'])
def training_data():
    return jsonify(db_get_all_training())


@app.route('/testing-data', methods=['GET', 'POST'])
def testing_data():
    return jsonify(db_get_all_testing())


@app.route('/import-training', methods=['GET', 'POST'])
def import_training():
    file = request.files['file']
    file_name = secure_filename(file.filename)
    file_path = 'Upload/' + file_name
    file.save(file_path)
    db_delete_all("training")
    db_import_data_training(file_path)
    return file_path


@app.route('/import-testing', methods=['GET', 'POST'])
def import_testing():
    file = request.files['file']
    file_name = secure_filename(file.filename)
    file_path = 'Upload/' + file_name
    file.save(file_path)
    db_delete_all("testing")
    db_import_data_testing(file_path)
    return file_path


@app.route('/halaman-pengujian', methods=['GET', 'POST'])
def halaman_proses_pengujian():
    data_testing = db_get_all_testing()
    data_training = db_get_all_training()
    data = {
        "data_testing": data_testing,
        "data_training": data_training,
        # "tf_idf" :
    }
    return jsonify(data)


@app.route('/pengujian', methods=['GET', 'POST'])
def pengujian():
    # proses tf-idf training dan testing
    tfidf(get_hasil())
    # read csv hasil tf-idf
    df = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')
    # membagi hasil tf idf dari total data set ke 600 training, 400 testing
    split1 = (df.shape[0] * 6) / 10
    # baca hasil split tf-idf data training dan testing
    df1 = pd.DataFrame(df.iloc[:int(split1)])
    df2 = pd.DataFrame(df.iloc[int(split1):])
    # proses jarak euclidean dan menyimpan data ke dalam file csv
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)
    # read csv jarak_dtdt dan jarak_dtds
    jarak_dtdt = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=',')
    jarak_dtds = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=',')
    # mengurutkan data jarak_dtdt
    small_dtdt = small(jarak_dtdt)
    # jumlah k yang digunakan
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
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
            lbl2 = []
            for i in label:
                lbl2.append(label_train[i])
            labelterdekat = pd.DataFrame(lbl2)
            labelterdekat.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/labelterdekat.json')
            labelterdekat.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelterdekat.csv')
            lbl.append(most_frequent(lbl2))
            updateData(nama_polaritas, most_frequent(lbl2), idx)
            idx += 1
    status = "success"
    return status


@app.route('/test-akurasi', methods=['GET', 'POST'])
def test_akurasi():
    cm_list = []
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for i in k:
        cm_list.append(testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(i)))
    return jsonify(cm_list)


@app.route('/halaman-user', methods=['GET', 'POST'])
def halaman_user():
    return jsonify(get_all_user())


@app.route('/add-user-proses', methods=['GET', 'POST'])
def add_user_proses():
    nama = request.form.get("nama")
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")
    data = add_user(nama, username, email, password)
    if bool(data):
        response = {
            'status': 'success'
        }
    else:
        response = {
            'status': 'failed'
        }
    return response


@app.route('/halaman-updateu', methods=['GET', 'POST'])
def halaman_update_user():
    id = request.form.get("id")
    data = get_user_by_id(id)
    if bool(data):
        response = {
            'data': data
        }
    else:
        response = {
            'data': data
        }
    return response


@app.route('/update-user-proses', methods=['GET', 'POST'])
def update_user_proses():
    id = request.form.get("id")
    nama = request.form.get("nama")
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")
    data = update_user(id, nama, username, email, password)
    if bool(data):
        response = {
            'status': 'success'
        }
    else:
        response = {
            'status': 'failed'
        }
    return response


@app.route('/dell-user', methods=['GET', 'POST'])
def dell_user():
    id = request.form.get("id")
    data = db_dell_user(id)
    if data == "success":
        response = {
            'status': 'success'
        }
    else:
        response = {
            'status': 'failed'
        }
    return jsonify(response)


@app.route('/dell-all', methods=['GET', 'POST'])
def dell_all():
    tabel = request.form.get("tabel")
    data = db_delete_all(tabel)
    if data == "success":
        response = {
            'status': 'success'
        }
    else:
        response = {
            'status': 'failed'
        }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host=my_ip)