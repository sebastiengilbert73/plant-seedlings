# Loader.py
import argparse
import torch
import torchvision
import PIL
import io
import os
import random
import numpy


"""class SampleImage:
    def __init__(self, imageID, tensor, label=None):
        self.imageID = imageID
        self.tensor = tensor
        self.label = label
"""

class Importer:
    def __init__(self, categoriesDirectory, maximumNumberOfImages=0):
        # List the categories in the categoriesDirectory
        directoriesList = [os.path.join(categoriesDirectory, o) for o in os.listdir(categoriesDirectory)
                           if os.path.isdir(os.path.join(categoriesDirectory, o))]
        #print ("Import.__init__(): directoriesList =", directoriesList)
        self.filepathToClassDic = {}
        self.filepaths = []
        self.classNames = []
        for directory in directoriesList:
            className = os.path.basename(directory)
            self.classNames.append(className)
            filepaths = [os.path.join(directory, f) for f in os.listdir(directory)
                         if os.path.isfile(os.path.join(directory, f))]
            for filepath in filepaths:
                self.filepathToClassDic[filepath] = className
                self.filepaths.append(filepath)
        print("")

    def ClassIndex(self, className):
        for classNdx in range(len(self.classNames)):
            if self.classNames[classNdx] == className:
                return classNdx
        raise Exception("Importer.ClassIndex(): Class name {} was not found in the list".format(className))

    def ClassName(self, classNdx):
        return self.classNames[classNdx]

    def FilepathToClassDic(self):
        return self.filepathToClassDic

    def SplitForTrainAndValidation(self, numberOfTrainingImages, numberOfValidationImages):
        trainFilepathToClassDic = {}
        validationFilepathToClassDic = {}
        correctedNumberOfTrainingImages = numberOfTrainingImages
        correctedNumberOfValidationImages = numberOfValidationImages
        if numberOfTrainingImages + numberOfValidationImages > len(self.filepathToClassDic):
            correctedNumberOfTrainingImages = int (numberOfTrainingImages * len(self.filepathToClassDic) / \
                                    (numberOfTrainingImages + numberOfValidationImages) )
            correctedNumberOfValidationImages = len(self.filepathToClassDic) - correctedNumberOfTrainingImages
        # Generate a list of random indices
        indices = random.sample(range(len(self.filepaths)), len(self.filepaths))
        for trainNdx in range(correctedNumberOfTrainingImages):
            index = indices[trainNdx]
            filepath = self.filepaths[index]
            trainFilepathToClassDic[filepath] = self.filepathToClassDic[filepath]
        for validationNdx in range(correctedNumberOfTrainingImages, correctedNumberOfTrainingImages + correctedNumberOfValidationImages):
            index = indices[validationNdx]
            filepath = self.filepaths[index]
            validationFilepathToClassDic[filepath] = self.filepathToClassDic[filepath]
        return trainFilepathToClassDic, validationFilepathToClassDic

    def Preprocessing(self, imageSize):
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize((imageSize[1], imageSize[0])),  # Resize expects (h, w)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ])
        return preprocessing

    def ConvertToTensors(self, filepathToClassDic, imageSize): # imageSize: (width, height)
        imagesTensor = torch.FloatTensor(len(filepathToClassDic), 3, imageSize[1], imageSize[0]) # NCHW
        labelsTensor = torch.LongTensor(len(filepathToClassDic), 1)

        preprocessing = self.Preprocessing(imageSize)
        filepathNdx = 0
        for filepath in filepathToClassDic:
            pilImg = PIL.Image.open(filepath)
            #pilImg.show()
            imgTensor = preprocessing(pilImg)
            tensorShape = imgTensor.shape
            if imgTensor.shape[0] > 3:
                imgTensor = imgTensor[0:3]
            #if tensorShape[0] == 3:

            imagesTensor[filepathNdx] = imgTensor

            className = filepathToClassDic[filepath]
            classNdx = self.ClassIndex(className)
            labelsTensor[filepathNdx] = classNdx
            filepathNdx += 1
            #else:
            #    print ("Importer.ConvertToTensors(): The image {} doesn't have 3 channels".format(filepath))


        return imagesTensor, labelsTensor

    def MinibatchIndices(self, numberOfSamples, minibatchSize):
        shuffledList = numpy.arange(numberOfSamples)
        numpy.random.shuffle(shuffledList)
        minibatchesIndicesList = []
        numberOfWholeLists = int(numberOfSamples / minibatchSize)
        for wholeListNdx in range(numberOfWholeLists):
            minibatchIndices = shuffledList[wholeListNdx * minibatchSize: (wholeListNdx + 1) * minibatchSize]
            minibatchesIndicesList.append(minibatchIndices)
        # Add the last incomplete minibatch
        if numberOfWholeLists * minibatchSize < numberOfSamples:
            lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
            minibatchesIndicesList.append(lastMinibatchIndices)
        return minibatchesIndicesList

# Tester
if __name__ == '__main__':
    print ("Loader.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('baseDirectory', help='The directory containing the directories train/ and test/')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    importer = Importer(os.path.join(args.baseDirectory, 'train'))
    filepathToClassDic = importer.FilepathToClassDic()
    #print ("filepathToClassDic =", filepathToClassDic)
    print ("len(filepathToClassDic) =",len(filepathToClassDic))
    trainFilepathToClassDic, validationFilepathToClassDic = importer.SplitForTrainAndValidation(1000, 250)
    trainImagesTensor, trainLabelsTensor = importer.ConvertToTensors(trainFilepathToClassDic, (256, 256))
    print ("trainLabelsTensor =", trainLabelsTensor)