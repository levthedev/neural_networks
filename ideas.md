* train nn on IMageNet, feedforwards images, find w and b, ff 2nd image find w and b, average w and b, deconv w and b to find average convultional pixel space between image1 and image2

* inception net - nn in nn, where outer nn acts like conv but instead of local receptive field being convultion, it is multilayer perceptron. what if instead of MLP, you used SVM, or even a nested convnet? if nested convnet, you'd have to make stride size large for outer and smaller for inner.

* dropout drops random activations, dropconnect drops random weights. what about randomizing some of the activation functions? either RELU -> linear, or tanh -> sigmoid, or maybe just shifting it to be a() +- 1, kind of like artificial training set enlargement.

* add frontend to handwritten digit nn so users can draw in numbers (and maybe even upload pics)

* chrome extension with trained nn (aka pre optimized weights/biases) that generates captions for images and places them as alt text on img tgs in html - see https://cs.stanford.edu/people/karpathy/deepimagesent/.

* nn that tries to autocomplete coding (or prose?) trained on github or guttenberg data sets - see html rnn post

* js library for batched (distributed) stochastic gradient descent - like karpathy's pre trained js library, but instead of pre trained, have clients act as distributed CPU's to do the training.

* nn that plays ssbm

* nn with homeomorphically encrypted data as test data (and maybe also training data if no consentual training data can be collected)

* machine learning papers usually compare the performance of algorithms to the performance of humans. but by humans, they mean able bodied humans. by cutting out differently abled people, machine learning researchers miss a number of possibilities for designing algorithms. for example, although neural networks that do object recognition are improving quickly, they still have trouble and oftentimes misidentify objects that most people would never misidentify. but there's the rub - most people is not all people. and so for example, a blind person would likely find immense utility in having an algorithm that could be triggered via his/her watch or smartphone that would analyze the scene in front of them (either a picture from a smartphone, glasses like google glass, or a separate camera module/wearable) and tell them what objects they see. the same idea applies to deaf people and speech to text neural networks, and mute people and text to speech neural networks.


why can't I find all points where dC/dw = 0 and dC/db = 0 and then of those pick lowest C to find absolute minima?
is it because I haven't defined the Cost function, and only have trial and error points of C for the weights and biases I've solved?
Cost(w, b) = 1/2n( sumForAllX( (y(x) - a)^2 ) )

need to learn more about chain rule with multi variable calculus - backpropagation is just applying chain rule with multi variable calculus starting on output layer and working your way backwards to beginning to calculate the gradient of the cost function (so as to minimize it)


input - 784 pixels -> conv -> maxpool -> conv -> maxpool -> fully connected -> fully connected -> output
train 10 networks as such:
* w and b initialization is normal distribution
* activation is ReLU
* cost function is log-likelihood
* final layer is softmax
* l2 regularization
* decreasing learning rate
* artificially expanding training data by shifting pixels +- (1..8) & rotating +- (1..5) degrees (gives total of 5mil images)
* dropout on last layer (all fully connected layers? maybe dropconnect?)

* have 10 networks average percentages and take best
