from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
parser.add_argument('--vocab_size', help='vocabullary size', type=int, default=400)
args = parser.parse_args()

DATA_PATH = '../data/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
              'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
              'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
              'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agri', 'fly', 'bball', 'beach', 'bldngs', 'chprl', 'dres',
                   'frst', 'fway', 'golf', 'hrbr', 'intr', 'mres', 'mblpark', 'opass',
                   'prkng', 'rvr', 'rway', 'sres', 'tanks', 'tennis']


# FEATURE = args.feature
FEATURE  = 'bag_of_sift'

# CLASSIFIER = args.classifier
CLASSIFIER = 'support_vector_machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 80
NUM_TEST_PER_CAT = 20


def main():
    vocab_sizes = [100, 200, 300, 400, 500]
    accuracies = []
    for vocab_size in vocab_sizes:
        #This function returns arrays containing the file path for each train
        #and test image, as well as arrays with the label of each train and
        #test image. By default all four of these arrays will be 1500 where each
        #entry is a string.
        print("Getting paths and labels for all train and test data")
        train_image_paths, test_image_paths, train_labels, test_labels = \
            get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT)

        # TODO Step 1:
        # Represent each image with the appropriate feature
        # Each function to construct features should return an N x d matrix, where
        # N is the number of paths passed to the function and d is the 
        # dimensionality of each image representation. See the starter code for
        # each function for more details.

        if FEATURE == 'tiny_image':
            # YOU CODE get_tiny_images.py 
            train_image_feats = get_tiny_images(train_image_paths)
            test_image_feats = get_tiny_images(test_image_paths)

        elif FEATURE == 'bag_of_sift':
            # YOU CODE build_vocabulary.py
            # vocab_size = args.vocab_size
            if os.path.isfile(f'vocab_{vocab_size}.pkl') is False:
                print('No existing visual word vocabulary found. Computing one from training images\n')
                # vocab_size = args.vocab_size   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                vocab = build_vocabulary(train_image_paths, vocab_size)
                with open(f'vocab_{vocab_size}.pkl', 'wb') as handle:
                    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if os.path.isfile(f'train_image_feats_1_{vocab_size}.pkl') is False:
                # YOU CODE get_bags_of_sifts.py
                train_image_feats = get_bags_of_sifts(train_image_paths, vocab_size)
                with open(f'train_image_feats_1_{vocab_size}.pkl', 'wb') as handle:
                    pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'train_image_feats_1_{vocab_size}.pkl', 'rb') as handle:
                    train_image_feats = pickle.load(handle)

            if os.path.isfile(f'test_image_feats_1_{vocab_size}.pkl') is False:
                test_image_feats  = get_bags_of_sifts(test_image_paths, vocab_size)
                with open(f'test_image_feats_1_{vocab_size}.pkl', 'wb') as handle:
                    pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'test_image_feats_1_{vocab_size}.pkl', 'rb') as handle:
                    test_image_feats = pickle.load(handle)
        elif FEATURE == 'dumy_feature':
            train_image_feats = []
            test_image_feats = []
        else:
            raise NameError('Unknown feature type')

        # TODO Step 2: 
        # Classify each test image by training and using the appropriate classifier
        # Each function to classify test features will return an N x 1 array,
        # where N is the number of test cases and each entry is a string indicating
        # the predicted category for each test image. Each entry in
        # 'predicted_categories' must be one of the 15 strings in 'categories',
        # 'train_labels', and 'test_labels.

        if CLASSIFIER == 'nearest_neighbor':
            # YOU CODE nearest_neighbor_classify.py
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

        elif CLASSIFIER == 'support_vector_machine':
            # YOU CODE svm_classify.py
            predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

        elif CLASSIFIER == 'dumy_classifier':
            # The dummy classifier simply predicts a random category for
            # every test case
            predicted_categories = test_labels[:]
            shuffle(predicted_categories)
        else:
            raise NameError('Unknown classifier type')

        accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
        print(f"Accuracy for vocab size {vocab_size}= ", accuracy)
        accuracies.append(accuracy)

        for category in CATEGORIES:
            accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
            print(str(category) + ': ' + str(accuracy_each))
        
        test_labels_ids = [CATE2ID[x] for x in test_labels]
        predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
        train_labels_ids = [CATE2ID[x] for x in train_labels]
        
        # Step 3: Build a confusion matrix and score the recognition system
        # You do not need to code anything in this section. 
        build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES, vocab_size)
        visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(vocab_sizes, accuracies, marker='o', linestyle='-', linewidth=2)

    # Add labels and title
    plt.xlabel("Vocab Size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy vs. Vocab Size", fontsize=16)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Optional: Annotate each point
    for x, y in zip(vocab_sizes, accuracies):
        plt.text(x, y, f"{y:.2f}", fontsize=10, ha='right')

    # Show or save the plot
    plt.savefig("accuracy_vs_vocab_size.png")  # Save the plot as a file

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories, vocab_size):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')
    plt.savefig(f"/home/vmulukuri/Seminar_Impl/GNR638_Assgn_1_Bag_of_Words_Classification/results/bag_of_sift-support_vector_machine_{vocab_size}.png", format = "png")
    plt.show()
    
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

if __name__ == '__main__':
    main()
