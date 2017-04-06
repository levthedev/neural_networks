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
