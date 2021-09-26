from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
import numpy as np
import timeit
from sklearn.model_selection import StratifiedShuffleSplit


def final_classifier_evaluation(classifier,X_train, X_test, y_train, y_test, multiclass):
    
    start_time = timeit.default_timer()
    classifier.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = classifier.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    if multiclass:
        #y_score = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(X_test)])
        auc = 0.00#roc_auc_score(y_test, y_score, multi_class='ovo', average="weighted")
        f1 = f1_score(y_test,y_pred, average="weighted")
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred, average="weighted")
        recall = recall_score(y_test,y_pred, average="weighted")
    else:
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)

    print("Evaluation on Test Dataset")
    print("*****************************************************")
    print("Training Time (s):   "+"{:.5f}".format(training_time))
    print("Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")


def get_data(dataset,index):

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=30)
    for train_index, test_index in split.split(dataset, dataset[index]):
        strat_train_set = dataset.loc[train_index]
        strat_test_set = dataset.loc[test_index]
    
    train_set = strat_train_set
    test_set = strat_test_set


    train_y = train_set[[index]]
    train_x= train_set.drop(index, axis=1)
    test_y = test_set[[index]]
    test_x = test_set.drop(index, axis=1)
    
    return train_x, train_y, test_x, test_y


def validate_and_plot(classifier, X, y, scoring_type, title):
    
    train_mean = []
    cv_mean = []
    train_time_mean = []
    pred_time_mean = []
    error = []
    mae_train = []
    mae_test = []
    train_sizes=(np.linspace(.05, 1.0, 20)*len(y)).astype('int64')  
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X.iloc[idx,:]
        y_subset = y.iloc[idx]
        scores = cross_validate(classifier, X_subset, y_subset, cv=10, scoring=scoring_type, n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score']))
        train_time_mean.append(np.mean(scores['fit_time']))
        pred_time_mean.append(np.mean(scores['score_time']))

        mean_error = cross_val_score(classifier, X_subset, y_subset, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        error.append(np.mean(-mean_error))

    kf = KFold(n_splits=20)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_test.append(mean_absolute_error(y_test, y_test_pred))
            
    train_mean = np.array(train_mean)
    cv_mean = np.array(cv_mean)
    train_time_mean = np.array(train_time_mean)
    pred_time_mean = np.array(pred_time_mean)
    mae_train = np.array(mae_train)
    mae_test = np.array(mae_test)

    folds = range(1, kf.get_n_splits() + 1)
    plot_graph("Learning Curve: "+title, "Training Examples", "F1 Score",[train_mean,cv_mean], ["Training Score","Cross-Validation Score"], train_sizes) 
    plot_graph("Error Curve: "+title, "Training Examples", "validation Error",[error], ["val error"], train_sizes)
    plot_graph("Error Curve: "+title, "Training Examples", "validation Error",[mae_train,mae_test], ["train error", "test error"],  folds)
    plot_graph("Modeling Time: "+title, "Training Examples", "Training Time (s)",[train_time_mean,pred_time_mean], ["Training Time (s)","Prediction Time (s)"],train_sizes)
    
    return train_sizes, train_mean, train_time_mean, pred_time_mean
    

def plot_graph(title, xlabel, ylabel,x_axis, x_legend, y_axis):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    color = ["b-o","r-o","g-o","m-o","k-o"]
    for i in range(len(x_axis)):
        plt.plot(y_axis, x_axis[i], color[i], label=x_legend[i])
    plt.legend(loc="best")
    plt.show()
    
    