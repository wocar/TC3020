
### Installing required libraries

    pip install -r requirements.txt

### Cleaning

First we must clean the datasets by running

    python clean.py

This will substitute

    Male = 1
    Female = 0

This will result in two new files:

    default_clean.txt
    genero_clean.txt


### Training and testing

Run

    python default.py
    python genero.py


This will train **TWO** models using gradient descent. The first one will use my implementation, the second model will be SciLearn's implementation. Training will stop when the threshold is reached.

    Threshold = 0.000001

Once completed it will print a classification report **FOR EACH** implementation
    
                      precision    recall  f1-score   support
    
             0.0       0.93      0.92      0.92      1010
             1.0       0.92      0.93      0.92       990
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    

And a confusion matrix using the scilearn library 

    Confusion Matrix
    
    [[926  84]
     [ 74 916]]

As well as the accuracy:

    Accuracy: 0.923%



###References:

    - https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
    - https://github.com/arseniyturin/logistic-regression
    - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
    - https://medium.com/@lope.ai/logistic-regression-from-scratch-in-python-d1e9dd51f85d
