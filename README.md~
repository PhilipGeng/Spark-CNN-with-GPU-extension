# CNN on Spark
This is a generalised CNN(convolutional neural network) model implemented on Spark-scala. The model is abstracted down to layer models which contains models like CL(Convolutional layer), SL(subsampling layer), FL(fully-connected layer) and OL(Output layer). You can build up your own network architecture according to your need.
<h5>package structure:</h5>
<pre>
./report.doc: Project report on detail methodologies and implementations<br>
./model: successfully trained LeNet5<br>
----+mlpmodel: trained mlp model<br>
----+localmodel: trained local model, sigmoid activated, 4 iterations, err<2%<br>
----+batch50: trained cluster model, use model of 1 round local training, totally 5 iterations, err<3%<br>
./src: source code of the package:<br>
----+CNNLayer<br>
    ----+layer.scala: generalized layer model with useful LA utilities (with learning rate and momentum)<br>
    ----+CL.scala: convolutional layer (sigmoid activation, easy to extend in tanh or arctan)<br>
    ----+SL.scala: subsampling layer (mean sampling, easy to extend to max/min sampling)<br>
    ----+FL.scala: fully connected layer<br>
    ----+OL.scala: extends FL, for output (sigmoid activation, trying to extend in softmax)<br>
----+CNNNet<br>
    ----+CNN.scala: well encapsulated LeNet5 model<br>
    ----+MLP.scala: well encapsulated multi-layer-perceptron model using CNNLayer<br>
----+NNFM<br>
    ----deprecated, stand alone experiment for Neural network in functional programming<br>
----+NNOO<br>
    ----deprecated, stand alone experiment for Neural Network in OO programming<br>
----+TestCase<br>
    ----+CNNClassifier<br>
        ----+cnnBatchTrain.scala: batch training of cnn<br>
        ----+cnnLocalTrain.scala: local model(single node) training of cnn<br>
        ----+cnnclassify.scala: feedforward operation of cnn<br>
        ----+cnntrain.scala: mixed training for cnn<br>
    ----+CNNCorrectnesss<br>
        ----+deprecated, for testing<br>
    ----+mlp<br>
	----+mlpclassify.scala<br>
	----+mlptrain.scala<br>
</pre>
<h5>Use language:</h5>
Spark (scala api)<br>
<h5>Based on theory:</h5>
Convolutional neural network trained with back propagation algorithm using gradient descent (parameter: learning rate and momentum).<br>
Global training setting: batch training size, training mode(update weight when classification is wrong or not?)
<h5>Supervised by:</h5>
Dr.Eric Lo, Hong Kong PolyU

