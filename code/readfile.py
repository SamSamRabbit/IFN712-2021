import numpy as np
import pandas as pd

# read training topic file
def readTopic(num):
    path = 'topic/Training' + str(num) + '.txt'
    with open(path) as f:
        lines = f.readlines()
        
    trainSet = []
    for line in lines:
        new_line = []
        new_line.append(line.split()[1])
        new_line.append(line.split()[2])
        trainSet.append(new_line)
    return trainSet

# read testing topic file
def readTopic2(num):
    path = 'topic/Test' + str(num) + '.txt'
    with open(path) as f:
        lines = f.readlines()
        
    trainSet = []
    for line in lines:
        new_line = []
        new_line.append(line.split()[1])
        new_line.append(line.split()[2])
        trainSet.append(new_line)
    return trainSet

# read training document
def readDocument(num1, num2):
    path = 'Training&Testing/Training' + str(num1) + '/' + num2 + '.txt'
    with open(path) as f:
        lines = f.readlines()
    return lines[0]

# read testing document
def readDocument2(num1, num2):
    path = 'Training&Testing/Testing' + str(num1) + '/' + num2 + '.txt'
    with open(path) as f:
        lines = f.readlines()
    return lines[0]