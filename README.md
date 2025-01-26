# Scene-recognition-with-bag-of-words

<center>
<img src="./README_files/header.png"><p style="color: #666;">
An example of a typical bag of words classification pipeline. Figure by <a href="http://www.robots.ox.ac.uk/~vgg/research/encoding_eval/">Chatfield et al.</a></p><p></p></center>

## Overview
The goal of this project is to solve classical computer vision topic, image recognition. In particular, I examine the task of scene recognition beginning with simplest method, tiny images and KNN(K nearest neighbor) classification, and then move forward to the state-of-the-art techniques, bags of quantized local features and linear classifiers learned by SVC(support vector classifier).

## Implementation
### 1. Vocabulary of Visual Words
After implementing a baseline scene recognition pipeline, we can finally move on to a more sophisticated image representation, bags of quantized SIFT features. Before we can represent our training and testing images as bag of feature histograms, we first need to establish a vocabulary of visual words. To create a vocabulary, we are going to sample several local feature based on SIFT descriptors, and then clustering them with kmeans. ```dsift(fast=True)``` is a efficient method to get SIFT descriptors, while ```kmeans()``` can return the cluster centroids. The number of clusters plays an important role, the larger the size, the better the performance. I set ```step_size=[5, 5]``` in order to accelerate the code.

NOTE: In this section, we have to run ```build_vocabulary.py```, which will take some time to construct the vocabulary.

```python3
bag_of_features = []

for path in image_paths:
img = np.asarray(Image.open(path),dtype='float32')
frames, descriptors = dsift(img, step=[5,5], fast=True)
bag_of_features.append(descriptors)
bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')

vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
```

### 2. Bag of SIFT
Now we are ready to represent our training and testing images as histograms of visual words. Theoretically, we will get a plenty of SIFT descriptors with ```dsift()``` function. Instead of storing hundreds of SIFT descriptors, we simply count how many SIFT descriptors fall into each cluster in our visual word vocabulary. We use euclidean distance to measure which cluster the descriptor belongs, creating corresponding histograms of visual words of each image. I have noticed that parameter  ```step``` varied accuracy quite a lot. I have tried with step=[5,5], step=[2,2] and step=[1,1]. Based on the experiment, the smaller the step, the higher the accuracy. It might because smaller step size can captere more details, contributing to more precise prediction. To avoid the wrong prediction due to various image size, I also normalized the histogram here.


### 3. SVMs(Support Vector Machines)
The last task is to train 1-vs-all linear SVMS to operate in the bag of SIFT feature space. Linear classifiers are one of the simplest possible learning models. The feature space is partitioned by a learned hyperplane and test cases are categorized based on which side of that hyperplane they fall on. ```LinearSVC()``` of scikit-learn provides a convenient way to implement SVMs. In addition, the parameter ```multi-class='ovr'``` realizes multi-class prediction. Hyperparameter tuning is extremely significant in this part, especially ```C```. I have tried with various value, from 1.0 to 5000.0, and the highest accuracy showed up on C=700.

```python
SVC = LinearSVC(C=700.0, class_weight=None, dual=True, fit_intercept=True,
intercept_scaling=1, loss='squared_hinge', max_iter= 2000,
multi_class='ovr', penalty='l2', random_state=0, tol= 1e-4,
verbose=0)

SVC.fit(train_image_feats, train_labels)
pred_label = SVC.predict(test_image_feats)
```

## Installation
1. Install [cyvlfeat](https://github.com/menpo/cyvlfeat) by running `conda install -c menpo cyvlfeat`
2. Run ```proj3.py```

Note: To tune the hyperparameter, please modify them directly in corresponding ```.py``` file, such as K(number of neighbors) in ```nearest_neighbor_classify.py```, C(penalty) in ```svm_classify```.

## Best Accuracy
```
Accuracy for vocab size 400=  0.7428571428571429
agricultural: 0.35
airplane: 0.5
baseballdiamond: 0.6
beach: 0.8
buildings: 0.8
chaparral: 1.0
denseresidential: 0.55
forest: 0.95
freeway: 0.85
golfcourse: 0.85
harbor: 1.0
intersection: 0.7
mediumresidential: 0.7
mobilehomepark: 0.9
overpass: 0.8
parkinglot: 0.95
river: 0.35
runway: 0.9
sparseresidential: 0.65
storagetanks: 0.6
tenniscourt: 0.8
```

## Results
Accuracy using Bag of sift features with linear SVM classifier reaches up to 0.742.

<table border=0 cellpadding=4 cellspacing=1>
<tr>
<th colspan=2>Confusion Matrix</th>
</tr>
<tr>
<td>Bag of SIFT ft. Linear SVM</td>
<td> 0.7428571428571429</td>
<td bgcolor=LightBlue><img src="results/bag_of_sift-support_vector_machine_400.png" width=400 height=300></td>
</tr>
</table>

## Visualization
| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen | ![](../data/train/Kitchen/image_0143.jpg) | ![](../data/test/Kitchen/image_0051.jpg) | ![](../data/test/TallBuilding/image_0197.jpg) | ![](../data/test/Kitchen/image_0111.jpg) |
| Store | ![](../data/train/Store/image_0298.jpg) | ![](../data/test/Store/image_0099.jpg) | ![](../data/test/Highway/image_0251.jpg) | ![](../data/test/Store/image_0090.jpg) |
| Bedroom | ![](../data/train/Bedroom/image_0143.jpg) | ![](../data/test/Bedroom/image_0215.jpg) | ![](../data/test/Coast/image_0338.jpg) | ![](../data/test/Bedroom/image_0016.jpg) |
| LivingRoom | ![](../data/train/LivingRoom/image_0149.jpg) | ![](../data/test/LivingRoom/image_0218.jpg) | ![](../data/test/Industrial/image_0305.jpg) | ![](../data/test/LivingRoom/image_0191.jpg) |
| Office | ![](../data/train/Office/image_0149.jpg) | ![](../data/test/Office/image_0183.jpg) | ![](../data/test/Coast/image_0356.jpg) | ![](../data/test/Office/image_0127.jpg) |
| Industrial | ![](../data/train/Industrial/image_0143.jpg) | ![](../data/test/Industrial/image_0262.jpg) | ![](../data/test/Mountain/image_0298.jpg) | ![](../data/test/Industrial/image_0245.jpg) |
| Suburb | ![](../data/train/Suburb/image_0157.jpg) | ![](../data/test/Suburb/image_0034.jpg) | ![](../data/test/Forest/image_0180.jpg) | ![](../data/test/Suburb/image_0053.jpg) |
| InsideCity | ![](../data/train/InsideCity/image_0143.jpg) | ![](../data/test/InsideCity/image_0060.jpg) | ![](../data/test/Highway/image_0029.jpg) | ![](../data/test/InsideCity/image_0084.jpg) |
| TallBuilding | ![](../data/train/TallBuilding/image_0071.jpg) | ![](../data/test/TallBuilding/image_0026.jpg) | ![](../data/test/OpenCountry/image_0064.jpg) | ![](../data/test/TallBuilding/image_0345.jpg) |
| Street | ![](../data/train/Street/image_0071.jpg) | ![](../data/test/Street/image_0186.jpg) | ![](../data/test/Mountain/image_0032.jpg) | ![](../data/test/Street/image_0083.jpg) |
| Highway | ![](../data/train/Highway/image_0143.jpg) | ![](../data/test/Highway/image_0017.jpg) | ![](../data/test/Coast/image_0088.jpg) | ![](../data/test/Highway/image_0006.jpg) |
| OpenCountry | ![](../data/train/OpenCountry/image_0143.jpg) | ![](../data/test/OpenCountry/image_0365.jpg) | ![](../data/test/Mountain/image_0126.jpg) | ![](../data/test/OpenCountry/image_0361.jpg) |
| Coast | ![](../data/train/Coast/image_0143.jpg) | ![](../data/test/Coast/image_0026.jpg) | ![](../data/test/OpenCountry/image_0037.jpg) | ![](../data/test/Coast/image_0243.jpg) |
| Mountain | ![](../data/train/Mountain/image_0349.jpg) | ![](../data/test/Mountain/image_0362.jpg) | ![](../data/test/Forest/image_0179.jpg) | ![](../data/test/Mountain/image_0126.jpg) |
| Forest | ![](../data/train/Forest/image_0143.jpg) | ![](../data/test/Forest/image_0060.jpg) | ![](../data/test/Mountain/image_0315.jpg) | ![](../data/test/Forest/image_0179.jpg) |

## Credits
This project is modified by Sai Vivek Mulukuri based on Chia-Hung Yuan which in turn is based on Min Sun, James Hays and Derek Hoiem's previous developed projects 
