#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random as rd

import matplotlib.pyplot as plt
# Taille de l'image
w = 800
h = 600

img = np.zeros((h, w, 3)) # image vide : que du noir

C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir

r = float(w) / h # rapport d'aspect
# coordonnées de l'écran : x0, y0, x1, y1. 
S = (-1., -1. / r , 1., 1. / r )

# Boucle sur l'ensemble des pixels 
for i, x in enumerate(np.linspace(S[0], S[2], w)):
	for j, y in enumerate(np.linspace(S[1], S[3], h)):
		couleur = (rd.random(),rd.random(),rd.random())
		img[h - j - 1, i] = couleur 

plt.imsave('figInit.png', img)
