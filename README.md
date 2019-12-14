# CNN-with-Machine-Learning
Using VGG16 to extract features from image to train ML model.

## Objective

Learn to use keras pretrained model to extract features from images and train Machine Learning model.

## Problem
Classify Fish and People from cifar-100 dataset.

## Dataset

Cifar-100:
This dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

Link: https://www.cs.toronto.edu/~kriz/cifar.html

## Steps

1. Load the dataset.
2. Extract People and Fish data from the dataset.
3. Reshape and Preprocess the images.
4. Load VGG16 model from keras using imagenet weights.
5. Extract Features from VGG16.
6. Train a ML model (we are using LogisticRegression)
7. Extract Features for test data
8. Test the model.

## Experiments

1. Tried using complete VGG16 for feautre extraction and used Logistic Regression for classification. <br>
    <b>Network Summary :-</b> <br>
    <pre>
    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param   
    =================================================================
    input_1 (InputLayer)         (None, 32, 32, 3)         0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    </pre>
    
   <b>Accuracy Score :</b> 0.871<br>
   <b>Classification Report:</b><br>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
       <th></th>
       <th>Label</th>
       <th>f1-score</th>
       <th>precision</th>
       <th>recall</th>
       <th>support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>Fish</td>
        <td>0.871129</td>
        <td>0.872000</td>
        <td>0.870259</td>
        <td>501.000</td>
      </tr>
      <tr>
        <th>1</th>
        <td>People</td>
        <td>0.870871</td>
        <td>0.870000</td>
        <td>0.871743</td>
        <td>499.000</td>
      </tr>
      <tr>
       <th>accuracy</th>
       <td></td>
       <td>0.871000</td>
       <td>0.871000</td>
       <td>0.871000</td>
       <td>0.871</td>
      </tr>
      <tr>
       <th>macro avg</th>
       <td></td>
       <td>0.871000</td>
       <td>0.871000</td>
       <td>0.871001</td>
       <td>1000.000</td>
      </tr>
      <tr>
       <th>weighted avg</th>
        <td></td>
        <td>0.871000</td>
        <td>0.871002</td>
        <td>0.871000</td>
        <td>1000.000</td>
      </tr>
    </tbody>
  </table>
<br>

2. Removed Last 8 layers of VGG16 and extracted features to train Logistic Regression.<br>
  <b>Network Summary:-</b>
  <pre>
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param   
    =================================================================
    block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4096)              0         
    =================================================================
    Total params: 1,735,488
    Trainable params: 1,735,488
    Non-trainable params: 0
    _________________________________________________________________
  </pre>
  
   <b>Accuracy Score :</b> 0.931
   <br>
   <b>Classification Report:</b>
  
  <table border="1" class="dataframe">
    <thead>
     <tr style="text-align: right;">
       <th></th>
       <th>Label</th>
       <th>f1-score</th>
       <th>precision</th>
       <th>recall</th>
       <th>support</th>
     </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>Fish</td>
        <td>0.930931</td>
        <td>0.930000</td>
        <td>0.931864</td>
        <td>499.000</td>
      </tr>
      <tr>
        <th>1</th>
        <td>People</td>
        <td>0.931069</td>
        <td>0.932000</td>
        <td>0.930140</td>
        <td>501.000</td>
      </tr>
      <tr>
        <th>accuracy</th>
        <td></td>
        <td>0.931000</td>
        <td>0.931000</td>
        <td>0.931000</td>
        <td>0.931</td>
      </tr>
      <tr>
        <th>macro avg</th>
        <td></td>
        <td>0.931000</td>
        <td>0.931000</td>
        <td>0.931002</td>
        <td>1000.000</td>
      </tr>
      <tr>
        <th>weighted avg</th>
        <td></td>
        <td>0.931000</td>
        <td>0.931002</td>
        <td>0.931000</td>
        <td>1000.000</td>
      </tr>
   </tbody>
  </table>

## Conclusion and Insights

The pretrained VGG16 model is trained on imagenet datasets which contains more than 20,000 categories.
So the last convolution layers will capture complex high level feature for those categories.
which is not required by our model.
Hence,the reduced model performed better for our dataset.
