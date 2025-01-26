from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pdb

def svm_classify(train_image_feats, train_labels, test_image_feats):
    #################################################################################
    # TODO :                                                                        #
    # This function will train a set of linear SVMs for multi-class classification  #
    # and then use the learned linear classifiers to predict the category of        #
    # every test image.                                                             # 
    #################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # LinearSVC :                                                                    #
    #   The multi-class svm classifier                                               #
    #        e.g. LinearSVC(C= ? , class_weight=None, dual=True, fit_intercept=True, #
    #                intercept_scaling=1, loss='squared_hinge', max_iter= ?,         #
    #                multi_class= ?, penalty='l2', random_state=0, tol= ?,           #
    #                verbose=0)                                                      #
    #                                                                                #
    #             C is the penalty term of svm classifier, your performance is highly#
    #          sensitive to the value of C.                                          #
    #   Train the classifier                                                         #
    #        e.g. classifier.fit(? , ?)                                              #
    #   Predict the results                                                          #
    #        e.g. classifier.predict( ? )                                            #
    ##################################################################################
    '''
    Input : 
        train_image_feats : training images features
        train_labels : training images labels
        test_image_feats : testing images features
    Output :
        Predict labels : a list of predict labels of testing images (Dtype = String).
    '''
    
    SVC = LinearSVC(C=700.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter= 2000,
                    multi_class='ovr', penalty='l2', random_state=0, tol= 1e-4,
                    verbose=0)
    
    # Initialize k-fold cross-validation
    k_folds = 8
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies = []

    print(f"Performing {k_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_image_feats)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # Split data into training and validation sets
        X_train, X_val = train_image_feats[train_idx], train_image_feats[val_idx]
        y_train, y_val = [train_labels[i] for i in train_idx], [train_labels[i] for i in val_idx]
        
        # Train the classifier on the current fold
        SVC.fit(X_train, y_train)
        
        # Predict validation labels
        val_predictions = SVC.predict(X_val)
        
        # Compute accuracy for the fold
        fold_accuracy = accuracy_score(y_val, val_predictions)
        accuracies.append(fold_accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {fold_accuracy:.4f}")

    # Report average validation accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"\nAverage Validation Accuracy: {avg_accuracy:.4f}")

    SVC.fit(train_image_feats, train_labels)
    
    pred_label = SVC.predict(test_image_feats)
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    
    return pred_label