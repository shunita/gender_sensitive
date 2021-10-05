# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:07:59 2021

@author: shunit.agmon
"""

def classify(embedding_file, label_prefix, model_name, model_desc,
             autoenc=None, emb_file_for_encoding=None,
             mode='enc_emb', use_idf_weights=False, test_label_prefix=None,
             cross_validation=True, read_cv_index=None, custom_test_file=None,
             filter_by_embedding_file=False, combine_mode='concat',
             random_size=None, big_dataset=True, pairs=None, return_model=False):
    # Embedding file female: "E:\\Shunit\\female1_emb.tsv"
    # label_name female: 'W_pos_dep'
    # label_prefix example: 'W_', 'M_', 'all_', 'W_>70_3_'
    # mode is one of: 'enc', 'emb', 'enc_emb', 'emb_random', 'emb_emb', 'emb_emb_enc'
    # combine_mode is one of 'concat', 'hadamard' and it affects what is done to the source and target vectors.
    # supported options for model_name: 'NN', 'LogisticRegression', 'XGBoost' 
    #1. read pairs and their label - sig/ not sig
    if pairs is None:
        if big_dataset:
            pairs = pd.read_csv(DISEASE_PAIRS_V2, index_col=0)
            pairs = pairs.rename({'source_name':'source', 'target_name':'target'}, axis=1)
        else:
            pairs = pd.read_csv(DISEASE_PAIRS_FILE, index_col=0)
    
    custom_test = None
    if custom_test_file is not None:
        custom_test = pd.read_csv(custom_test_file, index_col=0)
        print(f"read {len(custom_test)} rows from custom test file {custom_test_file}")
    
    label_name = label_prefix + 'pos_dep'
    if test_label_prefix is None:
        test_label = label_name
        test_label_prefix = label_prefix
    else:
        test_label = test_label_prefix + 'pos_dep'
    print("label:{} test_label: {}".format(label_name, test_label))
    if use_idf_weights:
        idfs = defaultdict(int, get_IDF_dict())
    
    if filter_by_embedding_file:
        emb_filter = read_embedding_file(embedding_file)
        before = len(pairs)
        pairs = pairs[pairs.apply(lambda row: (row['source_cui'] in emb_filter) and (row['target_cui'] in emb_filter), axis=1)]
        if mode.startswith("emb_emb"):
            emb_filter2 = read_embedding_file(emb_file_for_encoding)
            pairs = pairs[pairs.apply(lambda row: (row['source_cui'] in emb_filter2) and (row['target_cui'] in emb_filter2), axis=1)]
        print(f"filtered {len(pairs)} out of {before} that have embedding.")
    
    print("disease pairs to classify: {}".format(len(pairs)))
    
     #2. Get embeddings for the pairs
    if mode in ('emb', 'enc_emb', 'emb_random', 'emb_emb', 'emb_emb_enc'):
        emb = read_embedding_file(embedding_file)
        pairs['source_emb'] = pairs['source_cui'].apply(lambda x: emb[x])
        pairs['target_emb'] = pairs['target_cui'].apply(lambda x: emb[x])
        if custom_test is not None:
            custom_test['source_emb'] = custom_test['source_cui'].apply(lambda x: emb[x])
            custom_test['target_emb'] = custom_test['target_cui'].apply(lambda x: emb[x])
        print("finished reading embeddings")
    if mode in ('emb_emb', 'emb_emb_enc'):
        emb2 = read_embedding_file(emb_file_for_encoding)
        pairs['source_emb2'] = pairs['source_cui'].apply(lambda x: emb2[x])
        pairs['target_emb2'] = pairs['target_cui'].apply(lambda x: emb2[x])
        if custom_test is not None:
            custom_test['source_emb2'] = custom_test['source_cui'].apply(lambda x: emb2[x])
            custom_test['target_emb2'] = custom_test['target_cui'].apply(lambda x: emb2[x])
    # Get encoding for the pairs
    if mode in ('enc', 'enc_emb', 'emb_emb_enc'):
        if autoenc is None:
            print("Error: can't encode without autoencoder!")
            return
        # encode it now - autoencoder is not None at this point.
        if emb_file_for_encoding is not None:
            emb_for_enc = read_embedding_file(emb_file_for_encoding)
        else:
            emb_for_enc = read_embedding_file(embedding_file)
        pairs['source_emb_for_enc'] = pairs['source_cui'].apply(lambda x: emb_for_enc[x])
        pairs['target_emb_for_enc'] = pairs['target_cui'].apply(lambda x: emb_for_enc[x])
        if custom_test is not None:
            custom_test['source_emb_for_enc'] = custom_test['source_cui'].apply(lambda x: emb_for_enc[x])
            custom_test['target_emb_for_enc'] = custom_test['target_cui'].apply(lambda x: emb_for_enc[x])
        #w1 = w2 = pd.Series(1, index=pairs.index)
        if use_idf_weights:
            w1 = pairs['source_cui'].apply(lambda x: idfs[x])
            w2 = pairs['target_cui'].apply(lambda x: idfs[x])
            pairs['pair_emb'] = (pairs['source_emb_for_enc'].multiply(w1) + 
                 pairs['target_emb_for_enc'].multiply(w2)).divide(w1+w2)
            if custom_test is not None:
                w1 = custom_test['source_cui'].apply(lambda x: idfs[x])
                w2 = custom_test['target_cui'].apply(lambda x: idfs[x])
                custom_test['pair_emb'] = (custom_test['source_emb_for_enc'].multiply(w1) + 
                     custom_test['target_emb_for_enc'].multiply(w2)).divide(w1+w2)
        else:
            pairs['pair_emb'] = (pairs['source_emb_for_enc']+pairs['target_emb_for_enc'])/2
            if custom_test is not None:
                custom_test['pair_emb'] = (custom_test['source_emb_for_enc']+custom_test['target_emb_for_enc'])/2
        pair_emb_field_name = 'pair_emb'
        encoded_field_name = None
    if mode == 'emb_random':
        if random_size is None:
            print("random_size is None but mode is emb_random")
            return
        pairs['coded_rep'] = np.random.rand(len(pairs), random_size).tolist()
        if custom_test is not None:
            custom_test['coded_rep'] =  np.random.rand(len(custom_test), random_size).tolist()
        pair_emb_field_name = None
        encoded_field_name = 'coded_rep'
    else: # no encoding needed
        encoded_field_name = None
        encX_train = None
        encX_test = None
    
    #4. cross validation
    all_tests = None
    if not cross_validation:
        if read_cv_index is not None:
            # "train_test_indices151.pickle"
            train_index, test_index = pickle.load(open(read_cv_index, "rb"))
        else:
            train1, test1 = train_test_split(pairs, test_size=0.25)
            pickle.dump((train1.index, test1.index), open("train_test_indices.pickle", "wb"))
            CV_indices = [(train1.index, test1.index)]
        print("working in non-CV mode")
    else: # cross validation
        if read_cv_index is not None:
            CV_indices = pickle.load(open(read_cv_index, "rb"))
        else:
            CV_indices = list(ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(pairs))
            pickle.dump(CV_indices, open("CV_indices151_v2.pickle", "wb"))
        print("working in CV mode")
    accs = []
    aucs = []
    #fig, axes = plt.subplots(len(CV_indices), 1, figsize=(15,2), sharex=True)
    
    coefs = []
    #iteration = 0
    for train_index, test_index in CV_indices:
        train = pairs.loc[train_index]
        if custom_test_file is not None:
            test = custom_test
        else:
            test = pairs.loc[test_index]
        print("train index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
        train = train[(train[label_prefix+'1'] >= 30) & (train[label_prefix+'2'] >=30)]
        test = test[(test[test_label_prefix+'1'] >= 30) & (test[test_label_prefix+'2'] >=30)]
        print("after filter by number of patients with each disease:\ntrain index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
        
        if mode in ('enc', 'enc_emb', 'emb_random', 'emb_emb_enc'):
            before_enc_time = time.time()
            encX_train = encode_from_df(train, pair_emb_field_name, autoenc, encoded_field=encoded_field_name)
            encX_test = encode_from_df(test, pair_emb_field_name, autoenc, encoded_field=encoded_field_name)
            print("finished creating X and y and possibly encoding in {} secs".format(time.time()-before_enc_time))
        
        X_train, y_train = dataframe_to_x_and_y(train, label_name, encX_train, mode, combine_mode)
        X_test, y_test = dataframe_to_x_and_y(test, test_label, encX_test, mode, combine_mode)
        
        
        #5. Train classifier (logistic regression?)
        if model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=0)
#            model = LogisticRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pos_pred = (sum(pred)/len(pred))
            test['{}_pred_prob'.format(model_desc)] = model.predict_proba(X_test)[:, 1]
            coefs.append(np.abs(model.coef_))
#            im = axes[iteration].imshow(model.coef_)
#            axes[iteration].yaxis.set_ticks([])
#            axes[iteration].yaxis.set_ticklabels([])
#            
#            axes[iteration].xaxis.set_ticks([0, len(X_train[0])/2, len(X_train[0])])
        
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(objective="binary:logistic")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pos_pred = (sum(pred)/len(pred))
            test['{}_pred_prob'.format(model_desc)] = model.predict_proba(X_test)[:, 1]
            
        elif model_name == 'NN':
            input_size = X_train.shape[1]
            model = get_NN_model(input_size)
            model.fit(X_train, y_train, epochs=EPOCHS_FOR_NN_CLASSIFIER, batch_size=32, verbose=1)
            pred = model.predict(X_test)
            pos_pred = (sum(pred)/len(pred))[0]
            test['{}_pred_prob'.format(model_desc)] = pred
        
        test_readable = test[['source', 'target','source_cui', 'target_cui',
                              test_label, label_prefix+"1",
                              label_prefix+"2", label_prefix+"both", 
                              label_prefix+"none", label_prefix+"pval",
                              '{}_pred_prob'.format(model_desc)]]
        if all_tests is None:
            all_tests = test_readable
        else:
            all_tests = pd.concat([all_tests, test_readable])
        # how many positives?
        print("positives in train: {}, in test:{}, in prediction: {}".format(
                sum(y_train)/len(y_train),
                sum(y_test)/len(y_test),
                pos_pred))
        if model_name == 'NN':
            loss, acc = model.evaluate(X_test, y_test)
            auc = roc_auc_score(y_test, pred)
        else:
            acc = model.score(X_test, y_test)
            # print(model.classes_)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])     
        accs.append(acc)
        aucs.append(auc)
        #iteration+=1
    print("avg acc: {}, avg auc: {}, desc: {}".format(
            np.mean(accs),
            np.mean(aucs),
            '{}_pred_prob'.format(model_desc)))
    if model_name == 'LogisticRegression':
        avg_coefs = np.squeeze(np.mean(np.stack(coefs), axis=0))
        plt.bar(range(len(avg_coefs)), avg_coefs)
        #fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()
    if return_model:
        return model
    return accs, aucs, all_tests