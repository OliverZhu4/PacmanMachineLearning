import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            converge = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) == nn.as_scalar(y):
                    continue
                else:
                    converge = False
                    self.w.update(x, nn.as_scalar(y))

            if converge:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.lr = 0.005
        self.batch = 1
        self.convergeLoss = 0.02

        size = 100
        self.m1 = nn.Parameter(1, size)
        self.b1 = nn.Parameter(self.batch, size)
        self.m2 = nn.Parameter(size, 1)
        self.b2 = nn.Parameter(self.batch, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        l1 = nn.Linear(x, self.m1)
        b1 = nn.AddBias(l1, self.b1)
        r1 = nn.ReLU(b1)
        l2 = nn.Linear(r1, self.m2)
        b2 = nn.AddBias(l2, self.b2)
        predicted_y = b2
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            totalLoss = 0
            n_data = 0
            for x, y in dataset.iterate_once(self.batch):
                loss = self.get_loss(x, y)
                parameters = [self.m1, self.m2, self.b1, self.b2]
                gradients = nn.gradients(loss, parameters)
                for i in range(len(parameters)):
                    parameters[i].update(gradients[i], -self.lr)
                n_data += self.batch
                totalLoss += nn.as_scalar(loss)

            if totalLoss / n_data < self.convergeLoss:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.005
        self.batch = 4
        self.converge_accuracy = 0.975
        self.m1 = nn.Parameter(784, 400)
        self.b1 = nn.Parameter(1, 400)
        self.m2 = nn.Parameter(400, 60)
        self.b2 = nn.Parameter(1, 60)
        self.m3 = nn.Parameter(60, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l1 = nn.Linear(x, self.m1)
        b1 = nn.AddBias(l1, self.b1)
        r1 = nn.ReLU(b1)
        l2 = nn.Linear(r1, self.m2)
        b2 = nn.AddBias(l2, self.b2)
        r2 = nn.ReLU(b2)
        l3 = nn.Linear(r2, self.m3)
        b3 = nn.AddBias(l3, self.b3)
        predicted_y = b3
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        iters = 0
        converge = False
        while not converge:
            for x, y in dataset.iterate_once(self.batch):
                iters += 1
                loss = self.get_loss(x, y)
                parameters = [self.m1, self.m2, self.b1, self.b2]
                gradients = nn.gradients(loss, parameters)
                for i in range(len(parameters)):
                    parameters[i].update(gradients[i], -self.lr)
                    # self.lr = self.lr / 1.5
                if iters % 300 == 0 and dataset.get_validation_accuracy() > self.converge_accuracy:
                    converge = True
                    break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.003
        self.batch = 4
        self.converge_accuracy = 0.84
        size = 200
        self.m1 = nn.Parameter(self.num_chars, size)
        self.m2 = nn.Parameter(size, size)
        self.m3 = nn.Parameter(size, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l = nn.Linear(xs[0], self.m1)
        l = nn.ReLU(l)
        for i in range(1, len(xs)):
            l = nn.Add(nn.Linear(xs[i], self.m1), nn.Linear(l, self.m2))
            l = nn.ReLU(l)
        l = nn.Linear(l, self.m3)
        predicted_y = l
        return predicted_y

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(xs), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        iters = 0
        converge = False
        while not converge:
            for x, y in dataset.iterate_once(self.batch):
                iters += 1
                loss = self.get_loss(x, y)
                parameters = [self.m1, self.m2, self.m3]
                gradients = nn.gradients(loss, parameters)
                for i in range(len(parameters)):
                    parameters[i].update(gradients[i], -self.lr)
                if iters % 300 == 0 and dataset.get_validation_accuracy() > self.converge_accuracy:
                    converge = True
                    break
