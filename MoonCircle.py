# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:51:13 2018

@author: jean
"""

#  ----------------------------------
# | data                             |
#  ----------------------------------

FilePath = ''

#  ----------------------------------
# | import                           |
#  ----------------------------------

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
# from tkinter import *
from tkinter import filedialog

#  ----------------------------------
# | function                         |
#  ----------------------------------

def fitcircle(xin, yin):
    x = np.asarray(xin, dtype=float)
    y = np.asarray(yin, dtype=float)
    N = x.size

    u = np.ones(N)
    A = np.vstack((x, y, u)).T
    B = x**2 + y**2

    X, res, rank, s = np.linalg.lstsq(A, B, rcond=None)

    xc = X[0] / 2
    yc = X[1] / 2
    r  = np.sqrt(X[2] + xc**2 + yc**2)

    return xc, yc, r

def myhistogram(a):
    bins = np.linspace(-0.5, 255.5, 257)
    h, b = np.histogram(a, bins=bins)
    return h, b

def zoom_factory(ax, scale = 1.4):
    def zoom_fun(event):
        xleft, xright = ax.get_xlim()
        yinf,  ysup   = ax.get_ylim()
        x0 = event.xdata
        y0 = event.ydata
        if event.button == 'up':
            c = 1 / scale
        elif event.button == 'down':
            c = scale
        else:
            c = 1
        newxleft  = x0 - (x0 - xleft)  * c
        newxright = x0 + (xright - x0) * c
        newyinf   = y0 - (y0 - yinf)   * c
        newysup   = y0 + (ysup - y0)   * c
        ax.set_xlim([newxleft, newxright])
        ax.set_ylim([newyinf,  newysup])
        plt.draw()

    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event',zoom_fun)
    return zoom_fun

#  ----------------------------------
# | program                          |
#  ----------------------------------

# ------ ask image ------
Answer = filedialog.askopenfile(mode='r',
                                title='Choisir une image',
                                initialdir=FilePath,
                                filetypes=(('images', '*.jpg'), ('all files', '*.*')))

# ------ read image ------
FullName = Answer.name
print(FullName)
imc = plt.imread(FullName)
FileName = Answer.name.split('/')[-1]

# ------ plot image ------
fig1 = plt.figure('Color Source')
fig1.clf()
ax = fig1.add_subplot(111)
plt.imshow(imc, origin='upper')

# ------ histogram ------
imr = imc[:, :, 0]
img = imc[:, :, 1]
imb = imc[:, :, 2]
imw = (imr + img + imb) / 3

hir, ber = myhistogram(imr)
hig, beg = myhistogram(img)
hib, beb = myhistogram(imb)
hiw, bew = myhistogram(imw)

fig2 = plt.figure('Histogram')
fig2.clf()
plt.plot(ber[:-1]+0.5, hir, 'r')
plt.plot(beg[:-1]+0.5, hig, 'g')
plt.plot(beb[:-1]+0.5, hib, 'b')
plt.plot(bew[:-1]+0.5, hiw, 'k')

fig3 = plt.figure('Histogram Log')
fig3.clf()
plt.plot(ber[:-1]+0.5, np.log10(hir), 'r')
plt.plot(beg[:-1]+0.5, np.log10(hig), 'g')
plt.plot(beb[:-1]+0.5, np.log10(hib), 'b')
plt.plot(bew[:-1]+0.5, np.log10(hiw), 'k')

# ------ normalize ------
minr = imr.min()
maxr = imr.max()
imrn = (imr - minr) / (maxr - minr)
ming = img.min()
maxg = img.max()
imgn = (img - ming) / (maxg - ming)
minb = imb.min()
maxb = imb.max()
imbn = (imb - minb) / (maxb - minb)

im = (imrn + imgn + imbn) / 3
fig4 = plt.figure('Pseudo Color Optim')
fig4.clf()
ax = fig4.add_subplot(111)
plt.imshow(im, origin='upper')
plt.colorbar()


# ------ install zoom ------
f = zoom_factory(ax, scale = 1.4)

# ------ get mouse clic ------
fig4 = plt.figure('Pseudo Color Optim')
xy = plt.ginput(n=-1, timeout=-1, show_clicks=True)
N = len(xy)
vx = [t[0] for t in xy]
vy = [t[1] for t in xy]

# ------ find contour ------
xc, yc, R = fitcircle(vx, vy)
D = R * 2

# ------ standard deviation ------
s = np.sqrt(np.sum((np.sqrt((vx - xc) ** 2 + (vy - yc) ** 2) - R) ** 2) / N)

# ------ plot circle ------
theta = np.linspace(0, 360, 361) * np.pi/180.
x = xc + R * np.cos(theta)
y = yc + R * np.sin(theta)
plt.plot(x, y, 'y')
plt.plot(xc, yc, '+m',  markersize=20)
plt.plot(vx, vy, '+r', markersize=12, markerfacecolor='none')
plt.title('%s : xc=%.1f yc=%.1f D=%.1f +/- %.1f pixels' % (FileName, xc, yc, D, s), fontsize=10)
fig1.canvas.draw()

# ------ print result ------
print('File : %s xc=%.1f yc=%.1f D=%.1f +/- %.1f pixels' % (FileName, xc, yc, D, s))
img = Image.open(FullName)
exif = {}
for key, val in dict(img._getexif()).items():
    exif[TAGS.get(key)] = val
    # if key != 37510 and key != 37500:
    #     print('%5d' % key, '%26s' % TAGS.get(key), val)
print(exif['DateTimeOriginal'], exif['SubsecTimeOriginal'])    
    
# EXIF TAGS : https://www.exiv2.org/tags.html

# (x - xc)^2 + (y - yc)^2 = R^2
# x^2 - 2 x xc + xc^2 + y^2 - 2 y yc + yc^2 = R^2
# - 2 x xc - 2 y yc - R^2 + xc^2 + yc^2 = - x^2 -  y^2
# u x + v y + w = x^2 + y^2
# u = 2 xc
# v = 2 yc
# w = R^2 - xc^2 - yc^2

# u, v, w sont les inconnues
# [ . . . ] [ u ]   [        ]
# [ x y 1 ] [ v ] = [ x^2+y^2]
# [ . . . ] [ w ]   [        ]

# soit A * X = B
