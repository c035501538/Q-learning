import cv2
from cv2 import FONT_HERSHEY_PLAIN
from cv2 import FONT_HERSHEY_DUPLEX

text1 = 'waving hand'
text2 = 'passing'

image = cv2.imread('/home/sirius/RL練習/Q-learning/6.jpg')
image = cv2.resize(image, (300, 402), interpolation=cv2.INTER_AREA)
'''cv2.putText(image, 'waving hand', (0, 0), cv2,FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)'''
cv2.imshow('Result', image)
cv2.imwrite('/home/sirius/RL練習/Q-learning/66.jpg', image)
cv2.waitKey(0)