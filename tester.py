# tester.py
import argparse
import torch
import torchvision
import PIL
import io
import os
import random
import numpy
import math
import Loader
import ConvStackClassifier


if __name__ == '__main__':
    print ("tester.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('baseDirectory', help='The directory containing the directories train/ and test/')
    parser.add_argument('neuralNetworkFilename', help='The neural network filename')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--architecture', help='The neural network architecture (Default: ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.75)',
                        default='ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.75')
    parser.add_argument('--learningRate', help='The learning rate (Default: 0.001)', type=float, default=0.001)
    parser.add_argument('--momentum', help='The learning momentum (Default: 0.9)', type=float, default=0.9)
    parser.add_argument('--minibatchSize', help='The minibatch size (Default: 32)', type=int, default=32)


    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    loader = Loader.Importer(os.path.join(args.baseDirectory, 'train'))
    testDirectory = os.path.join(args.baseDirectory, 'test')
    imageFilepaths = [os.path.join(testDirectory, f) for f in os.listdir(testDirectory)
                         if os.path.isfile(os.path.join(testDirectory, f))]
    #print ("imageFilepaths =", imageFilepaths)

    # Create a neural network and an optimizer
    if args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_256_0.5':
        structureElements = ConvStackClassifier.ExtractStructureFromFilename(args.architecture)
        neuralNet = ConvStackClassifier.NeuralNet(structureElements[2], structureElements[0],
                                                  structureElements[3], structureElements[4],
                                                  structureElements[5])
        imageSize = (256, 256)
        # Load the neural network weights
        neuralNet.Load(args.neuralNetworkFilename)
        # Create an optimizer
        optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)
    elif args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.5' or \
            args.architecture == 'ConvStack_3_3_32_7_2_32_7_2_32_7_2_12_128_0.75':
        structureElements = ConvStackClassifier.ExtractStructureFromFilename(args.architecture)
        neuralNet = ConvStackClassifier.NeuralNet(structureElements[2], structureElements[0],
                                                  structureElements[3], structureElements[4],
                                                  structureElements[5])
        imageSize = (128, 128)
        # Load the neural network weights
        neuralNet.Load(args.neuralNetworkFilename)
        # Create an optimizer
        optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)

    else:
        raise NotImplementedError("tester.py: Unsupported architecture '{}'".format(args.architecture))

    if args.cuda:
        neuralNet.cuda()  # Move to GPU

    preprocessing = loader.Preprocessing(imageSize)
    numberOfMinibatches = math.ceil(len(imageFilepaths) / args.minibatchSize)
    print ("len(imageFilepaths) = {}; numberOfMinibatches = {}".format(len(imageFilepaths), numberOfMinibatches))

    submissionFile = open("submission_" + args.neuralNetworkFilename + '.csv', 'w')
    submissionFile.write('file,species\n')

    for minibatchNdx in range(numberOfMinibatches):
        startNdx = args.minibatchSize * minibatchNdx
        filepathIndices = list(range(startNdx, min(startNdx + args.minibatchSize, len(imageFilepaths)) ) )
        #print ("filepathIndices:", filepathIndices)

        imagesTensor = torch.FloatTensor(len(filepathIndices), 3, imageSize[1], imageSize[0]) # NCHW
        for filepathNdx in filepathIndices:
            imageFilepath = imageFilepaths[filepathNdx]
            #print (imageFilepath)
            pilImg = PIL.Image.open(imageFilepath)
            imgTensor = preprocessing(pilImg)
            tensorShape = imgTensor.shape
            if imgTensor.shape[0] > 3:
                imgTensor = imgTensor[0:3]

            imagesTensor[filepathNdx - startNdx] = imgTensor

        # Neural network forward pass
        prediction = torch.nn.functional.softmax(neuralNet( torch.autograd.Variable(imagesTensor, requires_grad=False)))
        #print (prediction)
        confidenceLevels, classIndices = prediction.max(1)
        for imageNdx in range(classIndices.data.shape[0]):
            filepathNdx = startNdx + imageNdx
            imageFilepath = imageFilepaths[filepathNdx]
            imageFilename = os.path.basename(imageFilepath)
            submissionFile.write(imageFilename + ',' + loader.ClassName(classIndices[imageNdx].data[0]) + '\n')
            print ("{}, {}".format(classIndices[imageNdx].data[0], confidenceLevels[imageNdx].data[0]))
    submissionFile.close()