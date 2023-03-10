#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_Ray(O, D):
    return { 
            'origine': O,
            'direction': D
            }

def create_Sphere(P, r, a, d, s, rf, i):
    return {
            'type': "sphere",
            'position': P,
            'rayon': r,
            'ambient': a,
            'diffuse': d,
            'specular': s,
            'reflection': rf,
            'index': i
            }
    
def create_Plane(P, n, a, d, s, r, i):
    return {
            'type': "plane",
            'position': P,
            'normale': n,
            'ambient': a,
            'diffuse': d,
            'specular': s,
            'reflection': r,
            'index': i
            }

def normalize(x):
    return x / np.linalg.norm(x)

def rayAt(ray,t):
    return ray['origine'] + t * ray['direction']

def get_Normal(obj, M):
    if obj['type'] == "sphere":
        normal = normalize(M - obj['position'])
    elif obj['type'] == "plane":
        normal = obj['normale']
    return normal 

def intersect_Plane(ray, plane):
    scal = np.dot(ray['direction'], plane['normale'])
    if np.abs(scal) < 1e-6:
        # pas d'intersection
        return np.inf
    distance = np.dot(plane['position'] - ray['origine'], plane['normale']) / scal
    if distance < 0:
        # pas d'intersection
        return np.inf
    return distance

def intersect_Sphere(ray, sphere):
    a = np.dot(ray['direction'], ray['direction'])
    dif = ray['origine'] - sphere['position'] 
    b = 2 * np.dot(ray['direction'], dif)
    c = np.dot(dif, dif) - sphere['rayon']**2
    discriminant = b**2 - 4 * a * c
    if discriminant > 0:
        racineDisc = np.sqrt(discriminant)
        v = (-b - racineDisc) / 2.0 if b < 0 else (-b+racineDisc) / 2.0
        p0 = v / a
        p1 = c / v
        p0, p1 = min(p0, p1), max(p0, p1)
        if p1 >= 0:
            if p0 < 0:
                return p1
            else:
                return p0
    return np.inf

def intersect_Scene(ray, obj):
    if obj['type'] == 'plane':
        return intersect_Plane(ray, obj)
    elif obj['type'] == 'sphere':
        return intersect_Sphere(ray, obj)

def Is_in_Shadow(obj_min,P,N):
    global L
    global scene
    global acne_eps

    liste = [] 
    lPl = normalize(L-P) 
    Pe = P + np.multiply(N, acne_eps)
    ray_test = create_Ray(Pe, lPl)
    for i, obj_test in enumerate(scene):
        if obj_test.get('index') != obj_min:
            inter = intersect_Scene(ray_test, obj_test)
            if inter != np.inf:
                liste.append(obj_test)
    if len(liste) == 0:
        return True
    else:
        return False


def eclairage(obj,light,P) : 
    global L
    global C
    global materialShininess

    couleur = obj['ambient'] * light['ambient']
    N = get_Normal(obj, P)
    l1 = max(np.dot(N,normalize(L - P)), 0)
    couleur += obj['diffuse'] * light['diffuse'] * l1
    l2 = max(np.dot(N, normalize( normalize( L - P ) + normalize(C - P) )), 0) ** (materialShininess / 4)
    couleur += obj['specular'] * light['specular'] * l2
    return couleur

def reflected_ray(dirRay,N):
    return dirRay['direction'] - 2 * np.dot(np.dot(dirRay['direction'], N), N)

def compute_reflection(rayTest,depth_max,col):
    global acne_eps

    d = rayTest['direction']
    c = 1
    depth = 0
    while depth < depth_max:
        trace = trace_ray(rayTest)
        if not trace:
            break
        obj, M, N, col_ray = trace
        rayTest = create_Ray(M+np.multiply(N, acne_eps), normalize(reflected_ray(rayTest, N)))
        col += c*col_ray
        depth += 1
        c *= obj['reflection']
    return col 

def trace_ray(ray):
    global Light
    global scene

    P = np.inf
    for i, obj_test in enumerate(scene):
        P_obj = intersect_Scene(ray, obj_test)
        if P_obj < P:
            P, obj_index = P_obj, i
    if P == np.inf:
        return None
    obj = scene[obj_index]
    M = rayAt(ray, P)
    N = get_Normal(obj, M)
    #shadow = Is_in_Shadow(obj, P, N)
    shadow = False
    if shadow: return None
    col_ray = eclairage(obj, Light, M)
    return obj, P, N, col_ray


# Taille de l'image
w = 800
h = 600
acne_eps = 1e-4
materialShininess = 50


img = np.zeros((h, w, 3)) # image vide : que du noir
#Aspect ratio
r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r , 1., 1. / r )


# Position et couleur de la source lumineuse
Light = { 'position': np.array([5, 5, 0]),
          'ambient': np.array([0.05, 0.05, 0.05]),
          'diffuse': np.array([1, 1, 1]),
          'specular': np.array([1, 1, 1]) }

L = Light['position']


col = np.array([0.2, 0.2, 0.7])  # couleur de base
C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir
materialShininess = 50
skyColor = np.array([0.321, 0.752, 0.850])
whiteColor = np.array([1,1,1])
depth_max = 10

scene = [create_Sphere([.75, -.3, -1.], # Position
                         .6, # Rayon
                         np.array([1. , 0.6, 0. ]), # ambient
                         np.array([1. , 0.6, 0. ]), # diffuse
                         np.array([1, 1, 1]), # specular
                         0.2, # reflection index
                         1), # index
          create_Plane([0., -.9, 0.], # Position
                         [0, 1, 0], # Normal
                         np.array([0.145, 0.584, 0.854]), # ambient
                         np.array([0.145, 0.584, 0.854]), # diffuse
                         np.array([1, 1, 1]), # specular
                         0.7, # reflection index
                         2), # index
         ]

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = col
        Q[:2] = (x, y)
        direc = normalize(Q-C)
        ray = create_Ray(C, direc)
        profondeur = 0

        while profondeur < depth_max:
            trace = trace_ray(ray)
            if not trace:
                break
            obj, P, N, col_ray = trace
            col *= col_ray
            profondeur += 1
        col = compute_reflection(ray, depth_max, col)
        
            
        img[h - j - 1, i, :] = np.clip(col, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]

plt.imsave('figRaytracing.png', img)
