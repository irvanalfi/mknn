import math

# Count term in word


def count_term(dict_text, text):
    for word in text:
        dict_text[word] += 1
    return dict_text


def comp_tf(doc_num_dict, doc):
    tf_doc = {}
    len_doc = len(doc)
    # every document
    for word, count in doc_num_dict.items():
        tf_doc[word] = count / float(len_doc)
    return tf_doc


def comp_df(documents: list):
    # initiate dictionary with first document
    df_dict = dict.fromkeys(documents[0], 0)
    # every word and value in doc in documents
    for doc in documents:
        for word, val in doc.items():
            # if dictionary word found +1 else initiate 1
            if word in df_dict:
                df_dict[word] += 1
            else:
                df_dict[word] = 1
    return df_dict


def comp_idf(documents: list):
    n = len(documents)

    idf_dict = {}
    df_dict = comp_df(documents)

    #  every word with value in df_dict calculate log(n/val)
    for word, val in df_dict.items():
        idf_dict[word] = math.log(n / float(val))
    return idf_dict


def comp_tf_idf(tfs, idfs):
    tfidf = {}
    for tf in tfs:
        for word, val in tf.items():
            tfidf[word] = val * idfs[word]
    return tfidf


def get_tf_idf(text_test: list, text_training_list: list):
    # Merge text to one list
    docs = [text_test, *text_training_list]

    # Get TF data and count term docs
    docs_num = []
    tfs = []
    for doc in docs:
        # change doc list to dictionary with initial 0 value
        doc_num = count_term(dict.fromkeys(doc, 0), doc)
        # add dictionary to list
        docs_num.append(doc_num)
        # calculate and get tf data from dictionary document
        tf = comp_tf(doc_num, doc)
        tfs.append(tf)

    # Get IDF Data
    idfs = comp_idf(docs_num)

    tfidfs = comp_tf_idf(tfs, idfs)
    # for tf in tfs:

    return tfidfs


def classification():
    pass
    # TODO


def testaccuracy():
    pass
    # TODO
