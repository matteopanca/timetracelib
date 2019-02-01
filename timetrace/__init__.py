#this can be an empty file
#https://docs.python.org/3/tutorial/modules.html#packages

import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication, QFileDialog
import numpy as np
import matplotlib.pyplot as plt
import sys

#----- Useful parameters -----
figsize_double = (18, 8)
figsize_single = (12, 8)
#-----------------------------

def fmr_kittel(h, m, n, unit='mT'):
	#m is the saturation magnetization in A/m
	#demag. factors, keeping in mind that n[2] is the applied field direction
	mu0 = np.pi*4.0e-7
	gamma = (2.0023*1.6021766e-19/(2*9.109383e-31))/(np.pi*2.0e9) #GHz/T
	if unit == 'Oe':
		h_conv = np.complex_(1e-4*h/mu0) #from Oe to A/m
	elif unit == 'mT':
		h_conv = np.complex_(1e-3*h/mu0) #from mT to A/m
	else:
		raise RuntimeError('Unit not yet supported')
	kittel_freq = mu0*gamma*np.real(np.sqrt((h_conv + (n[0] - n[2])*m)*(h_conv + (n[1] - n[2])*m)))
	return kittel_freq

def load_image(file_name, img_size=(5, 5)):
	image = plt.imread(file_name)
	fig = plt.figure(figsize=img_size)
	ax1 = fig.add_subplot(1,1,1)
	ax1.imshow(image)
	ax1.axis('off')
	fig.tight_layout()
	plt.show()

#Get the selected file's path in a string - TK
def get_path_tk(start_path='', filter=(('Data file', '*.dat'),('All files', '*'))):
	root = tk.Tk() #hide the root window
	root.withdraw() #hide the root window
	return filedialog.askopenfilename(initialdir=start_path, title='Select input file', filetypes=filter)

#Get the selected file's path in a string - QT
def get_path_qt(start_path='', filter='Data file (*.dat);;All files (*)'):
	app = QApplication(sys.argv)
	file_name = QFileDialog.getOpenFileName(None, 'Select input file', start_path, filter)
	return file_name[0]