import cv2
import mediapipe as mp

mp = mp.solutions.hands
hand = mp.Hands(max_num_hands=1)   
