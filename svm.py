import numpy as np

class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (in this homework, m will be 10)
        returns : numpy array of shape (n,m)
        """
        # step 1: initialize an array of shape (n_labels, m)
        n_labels = y.shape[0]
        ova_labels = np.full((n_labels, m), -1)
        
        # fill in the correct labels
        for i in range(n_labels):
            ova_labels[i, y[i]] = 1
        
        return ova_labels
        
    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : float
        self.w : (401, 10)
        """
        pred = x.dot(self.w)
        m = 10
        loss = .0
        for j in range(m):
            w_j = self.w[:,j]
            y_j = y[:,j]
            pred_j = pred[:,j]
            loss += self.C * np.mean(np.maximum(0, 1 - y_j * pred_j) ** 2) + .5 * np.linalg.norm(w_j) ** 2
        
        return loss
                  
    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """
        pred = x.dot(self.w)
        m = 10
        loss = .0
        
        n_features = x.shape[1]
        gradient = np.zeros((n_features,m))
        for j in range(m):
            w_j = self.w[:,j]
            y_j = y[:,j]
            pred_j = pred[:,j]
            active = (y_j * pred_j < 1).astype(float)
            gradient[:,j] = self.C * 2 * -((y_j * active)[:, np.newaxis] * x).mean(axis=0) + w_j
        return gradient
            
    
    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        preds = x.dot(self.w)
        n_examples = x.shape[0]
        m = 10
        y_inferred = np.full((n_examples, m), -1)
        most_likely_idx = np.argmax(preds, axis=1)
        for i in range(n_examples):
            y_inferred[i,most_likely_idx[i]] = 1
        
        return y_inferred

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        n_examples = y.shape[0]
        count_correct = np.sum(np.all(y_inferred == y, axis=1))
        return float(count_correct) / n_examples

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

    
if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    print("Fitting the model...")
    svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=False)
    train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    # svm.w = np.zeros([401, 10])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)