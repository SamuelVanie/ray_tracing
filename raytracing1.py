#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Taille de l'image
w = 800
h = 600

# image vide : que du noir
img = np.zeros((h, w, 3)) 
# Boucle sur l'ensemble des pixels 
for i in range(w):
	for j in range(h):
		# la couleur est un tuple modélisant les trois canaux Red, Green, Blue 
		couleur = (rd.random(),rd.random(),rd.random()) 
		img[h - j - 1, i] = couleur 

plt.imsave('figInit.png', img)