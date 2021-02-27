## Conclusions / Report

For me this activity was very enriching. 
It allowed me to understand the mathematical part behind the logistic regression algorithm, a very basic 
but yet powerful tool to have in our machine learning toolkit.

### Explanation

To train our model we need to find the correct weights (Beta) values that when multiplied by a given set (X) the result is Y

To do this we need to apply the logistic regression algorithm.

    while True:

First we calculate our hypothesis (h) multiplying our training set (X) by our weights (Beta) and pass this to the sigmoid function.

        anterior = self.beta
        h = self.__sigmoid(np.dot(X, self.theta))

Then we optimize our weights (B) by applying the gradient descent formula and calculate a new beta

        gradient = np.dot(X.T, (h - y)) / size
    
        self.beta = self.beta - (self.lr * gradient)

Then we calculate the difference between the current beta and the previous one

        delta = np.abs((anterior - self.beta)).mean()
    
If the threshold is reached then we end the loop

        if (np.abs(delta) < 0.000001): #Convergencia alcanzada
            break


### Results for genero.txt

With my implementation I was able to train the model with the following results

    Iterations: 38467 delta: 9.99983142055405e-07 loss: 0.20747303500919673 beta: [-0.45161771  0.18584126] lr = 5e-05


    Test results: 
                  precision    recall  f1-score   support
    
             0.0       0.92      0.92      0.92      1019
             1.0       0.91      0.92      0.92       981
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    
    Confusion Matrix
    
    [[935  84]
     [ 80 901]]
    
    Accuracy: 91.8%

Using the SciLearn implementation the program yielded the following results

    Test results: 
                  precision    recall  f1-score   support
    
             0.0       0.92      0.92      0.92      1019
             1.0       0.91      0.92      0.92       981
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    
    Confusion Matrix
    
    [[935  84]
     [ 81 900]]
    
    Accuracy: 91.75%

Accuracy was >90% so I think the model is accurate. Both SciLearn's and my implementation yielded very similar accuracy rates and test results

### Results for default.txt

With my implementation I was able to train the model with the following results

    Threshold reached: iterations: 1538 delta: 9.995425725694191e-07 loss: nan beta: [ 0.09617684  0.65743063 -0.05190119] lr = 5e-08

    
    Test results: 
                  precision    recall  f1-score   support
    
             0.0       0.98      0.94      0.96      1934
             1.0       0.18      0.39      0.25        66
    
        accuracy                           0.92      2000
       macro avg       0.58      0.67      0.60      2000
    weighted avg       0.95      0.92      0.93      2000
    
    Confusion Matrix
    
    [[1816  118]
     [  40   26]]
    
    Accuracy: 92.10000000000001%

Using the SciLearn implementation the program yielded the following results

    Training with SciLearn's LogisticRegression implementation.... 
    Test results: 
                  precision    recall  f1-score   support
    
             0.0       0.97      1.00      0.98      1934
             1.0       0.62      0.20      0.30        66
    
        accuracy                           0.97      2000
       macro avg       0.80      0.60      0.64      2000
    weighted avg       0.96      0.97      0.96      2000
    
    Confusion Matrix
    
    [[1926    8]
     [  53   13]]
    
    Accuracy: 96.95%

Even thought the accuracy is >90% in my implementation. 
By looking at the confusion matrix and at the test results, you can see:

                  precision    recall  f1-score   support
    
             0.0       0.98      0.94      0.96      1934
        ---> 1.0       0.18      0.39      0.25        66


However, if we look at the SciLearn's test results we see a much better result for Default=Yes

                  precision    recall  f1-score   support
    
             0.0       0.97      1.00      0.98      1934
             1.0       0.62      0.20      0.30        66


An 18% accuracy rate on guessing the Default=Yes, so im almost sure that my implementation is not accurate.

### Final comments

I don't know why I couldn't calculate the logloss value for the default.txt model. (It yielded nan)

## Personal reflections

### Parameters / Learning rate

I confirmed that the learning rate/iterations as well as the threshold are crucial parameters to 
successfully and quickly train our model.

### Dataset and biases

First I couldn't train the 'genero.txt' dataset, it would loop infinitely. 
Later I learned that I was not shuffling the dataset therefore having mostly males on the training set.

I solved this by randomizing the dataset with numpy.

### Showing the results

I learned about the confusion matrix and how to interpret it. 

Learned how to calculate the accuracy rate.

I would like to learn on how to plot the results to show something like this

![alt text](https://miro.medium.com/max/598/1*8uuWQtF0IHHUp6Ds3znL5Q.png)
https://medium.com/@lope.ai/logistic-regression-from-scratch-in-python-d1e9dd51f85d

### Frameworks and libraries

Practiced and learned a lot from the NumPy library to manipulate arrays and vectors. 

###References:

    - J, Guillermo Falcon. “Aprendizaje Automático. Clase: 2: el inicio del camino a Skynet” 2021. PDF file.
    - https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
    - https://github.com/arseniyturin/logistic-regression
    - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
    - https://medium.com/@lope.ai/logistic-regression-from-scratch-in-python-d1e9dd51f85d

