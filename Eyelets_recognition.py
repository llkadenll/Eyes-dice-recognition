import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from colorama import Fore, Style

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 0
params.filterByCircularity = True
params.minCircularity = 0.839
params.filterByConvexity = True
params.minConvexity = 0.97
params.filterByInertia = True
params.minInertiaRatio = 0.63

detector = cv2.SimpleBlobDetector_create(params)

numberOfImages = 38
correctCounter = 0
eyletsCorrectCounter = 0
blurValue = (16, 16)
imgs = []
properValues = [14, 6, 3, 6, 6, 10, 3, 6, 5, 14, 9, 13, 5, 6, 6, 15, 6, 10, 12,
               10, 14, 8, 10, 10, 10, 13, 10, 8, 5, 6, 14, 14, 8, 6, 8, 6, 6, 14]

def read_img(i):
    img = cv2.imread('dice\img_' + str(i) + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_img(i):
    plt.figure(figsize = (11, 11))
    plt.imshow(imgs[i])
    plt.show()

def eyelets_detection(i):  
    imgBlurred = cv2.blur(imgs[i], blurValue)
    return detector.detect(imgBlurred)

def dice_detection(eyelets):    
    blobCentroids = []
    for eye in eyelets:
        if eye.pt != None:
            blobCentroids.append(eye.pt)
            
    blobCentroids = np.asarray(blobCentroids)
    
    if len(blobCentroids) > 0:
        algorithm = cluster.DBSCAN(eps = 125, min_samples = 0).fit(blobCentroids)
        
        foundDice = max(algorithm.labels_) + 1
        dice = []
        
        for i in range(foundDice):
            diceCentroids = blobCentroids[algorithm.labels_ == i]
            centroid_dice = np.mean(diceCentroids, axis = 0)
            dice.append([len(diceCentroids), *centroid_dice])
        
        return dice
    else:
        return []
    
def draw_borders(i, dice, eyelets):
    for eyelet in eyelets:
        center = eyelet.pt
        leftUpper = tuple(int(i - (eyelet.size) / 2) for i in center)
        rightBottom = tuple(int(i + (eyelet.size) / 2) for i in center)
        cv2.rectangle(imgs[i], leftUpper, rightBottom, (0, 255, 0), 4)
        
    for d in dice:
        cv2.putText(imgs[i], str(d[0]), (int(d[1]), int(d[2])), 0, 2, (255, 0, 0), 7)


for i in range(numberOfImages):
    print(i + 1, '.')
    imgs.append(read_img(i + 1))
    eyelets = eyelets_detection(i)
    if len(eyelets) == properValues[i]:
        print('Wykryte oczka:', len(eyelets))
        print('Poprawna wartość:', properValues[i])
        correctCounter += 1
    else:
        print(Fore.RED + Style.BRIGHT + 'Wykryte oczka:', len(eyelets))
        print(Fore.RED + Style.BRIGHT + 'Poprawna wartość:', properValues[i])
    
    dice = dice_detection(eyelets)
    draw_borders(i, dice, eyelets)
    display_img(i) 

print("Bezbłędnie rozpoznano", round(correctCounter / numberOfImages * 100, 2), "% (", correctCounter, "/", numberOfImages, ")")