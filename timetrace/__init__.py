#this can be an empty file
#https://docs.python.org/3/tutorial/modules.html#packages

import tkinter as tk
from tkinter import filedialog

#----- Useful parameters -----
figsize_double = (18, 8)
figsize_single = (12, 8)
#-----------------------------

#Get the selected file's path in a string
def get_path(start_path='', filter=(('Data file', '*.dat'),('All files', '*'))):
	root = tk.Tk() #hide the root window
	root.withdraw() #hide the root window
	return filedialog.askopenfilename(initialdir=start_path, title='Select input file', filetypes=filter)