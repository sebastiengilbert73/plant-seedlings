import torch
import torchvision
import argparse
import ast
import os
import ConvStackClassifier
import Loader

print('trainer.py')

parser = argparse.ArgumentParser()
parser.add_argument('trainDirectory', help='The directory containing the class directories')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--architecture', help='The neural network architecture (Default: ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_256_0.5)', default='ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_256_0.5')
parser.add_argument('--restartWithNeuralNetwork', help='Restart the training with this neural network filename')
parser.add_argument('--lossFunction', help='Loss function (Default: CrossEntropyLoss)', default='CrossEntropyLoss')
parser.add_argument('--learningRate', help='The learning rate (Default: 0.001)', type=float, default=0.001)
parser.add_argument('--momentum', help='The learning momentum (Default: 0.9)', type=float, default=0.9)
parser.add_argument('--imageSize', help='The image size (width, height) (Default: (256, 256))', default='(256, 256)')
parser.add_argument('--numberOfEpochs', help='Number of epochs (Default: 200)', type=int, default=200)
parser.add_argument('--minibatchSize', help='Minibatch size (Default: 32)', type=int, default=32)
parser.add_argument('--numberOfValidationImages', help='The number of images used for validation (Default: 128)', type=int, default=128)
parser.add_argument('--numberOfTrainingImages', help='The maximum number of training images (Default: 0, which means no limit)', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

imageSize = ast.literal_eval(args.imageSize)

loader = Loader.Importer(args.trainDirectory, args.numberOfTrainingImages + args.numberOfValidationImages)
trainFilepathToClassDic, validationFilepathToClassDic = loader.SplitForTrainAndValidation(args.numberOfTrainingImages, args.numberOfValidationImages)
trainFilepaths = [*trainFilepathToClassDic]
validationFilepaths = [*validationFilepathToClassDic]
print ("len(trainFilepaths) = {}; len(validationFilepaths) = {}".format(len(trainFilepaths), len(validationFilepaths)))

# Create a neural network and an optimizer
if args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_256_0.5':
    structureElements = ConvStackClassifier.ExtractStructureFromFilename(args.architecture)
    neuralNet = ConvStackClassifier.NeuralNet(structureElements[2], structureElements[0],
                                              structureElements[3], structureElements[4],
                                              structureElements[5])
    imageSize = (256, 256)
    # Create an optimizer
    optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)
elif args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.5' or \
    args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.75':
    structureElements = ConvStackClassifier.ExtractStructureFromFilename(args.architecture)
    neuralNet = ConvStackClassifier.NeuralNet(structureElements[2], structureElements[0],
                                              structureElements[3], structureElements[4],
                                              structureElements[5])
    imageSize = (128, 128)
    # Create an optimizer
    optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)

else:
    raise NotImplementedError("trainer.py: Unsupported architecture '{}'".format(args.architecture))


validationImagesTensor, validationLabelsTensor = loader.ConvertToTensors(validationFilepathToClassDic, imageSize)
# Wrap validation data in Variables, once and for all
validationImagesTensor = torch.autograd.Variable(validationImagesTensor)
validationLabelsTensor = torch.autograd.Variable(validationLabelsTensor.squeeze(1))

if args.cuda:
    neuralNet.cuda()  # Move to GPU
    validationImagesTensor = validationImagesTensor.cuda()
    validationLabelsTensor = validationLabelsTensor.cuda()


# Create a loss function
if args.lossFunction == 'CrossEntropyLoss':
    lossFunction = torch.nn.CrossEntropyLoss()
else:
    raise NotImplementedError("trainer.py: Unsupported loss function '{}'".format(args.lossFunction))


#logSoftMax = torch.nn.LogSoftmax()

for epoch in range(1, args.numberOfEpochs + 1):
    print ("\nEpoch", epoch)
    averageTrainLoss = 0
    minibatchIndicesListList = loader.MinibatchIndices(len(trainFilepaths), args.minibatchSize)
    for minibatchListNdx in range(len(minibatchIndicesListList)):
        print('.', end="", flush=True)  # Print a dot without line return, right now
        minibatchIndicesList = minibatchIndicesListList[minibatchListNdx]
        thisMinibatchSize = len(minibatchIndicesList)
        minibatchFilepathToClassDic = {}
        for index in minibatchIndicesList:
            filepath = trainFilepaths[index]
            minibatchFilepathToClassDic[filepath] = trainFilepathToClassDic[filepath]

        minibatchImgsTensor, minibatchLabelsTensor = loader.ConvertToTensors(minibatchFilepathToClassDic, imageSize)

        # Wrap in Variable
        minibatchImgsTensor = torch.autograd.Variable(minibatchImgsTensor)
        minibatchLabelsTensor = torch.autograd.Variable(minibatchLabelsTensor.squeeze(1))
        if args.cuda:
            minibatchImgsTensor = minibatchImgsTensor.cuda()
            minibatchLabelsTensor = minibatchLabelsTensor.cuda()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        actualOutput = neuralNet(minibatchImgsTensor) # That's where the memory goes up

        # Loss
        #print (minibatchLabelsTensor)
        loss = lossFunction(actualOutput, minibatchLabelsTensor)
        #print ("loss =", loss)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        averageTrainLoss += loss.data[0]

    averageTrainLoss = averageTrainLoss / len(minibatchIndicesListList)
    # Validation
    validationOutput = neuralNet(validationImagesTensor)
    validationLoss = lossFunction(validationOutput, validationLabelsTensor)
    print("\nEpoch {}: Average train loss = {}; validationLoss = {}".format(epoch, averageTrainLoss, validationLoss.data[0]))
    torch.save(neuralNet.state_dict(), os.path.join('./', args.architecture + '_valLoss' + str(validationLoss.data[0]) ) )