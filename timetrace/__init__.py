#this can be an empty file
#https://docs.python.org/3/tutorial/modules.html#packages

import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication, QFileDialog
import matplotlib.pyplot as plt
import sys

#----- Useful parameters -----
figsize_double = (18, 8)
figsize_single = (12, 8)
#-----------------------------

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