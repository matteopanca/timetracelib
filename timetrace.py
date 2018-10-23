import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import tkinter as tk
from tkinter import filedialog

#----- Useful parameters -----
figsize_double = (18, 8)
figsize_single = (12, 8)
#-----------------------------

#----- Functions -----
def fit_expFunc(x, a, b, c, d, e):
	return (np.exp(-x/e)*a*np.cos(2*np.pi*b*(x+c)) + d)

def fit_coshFunc(x, a, b, c, d, e, f):
	return (a*np.cos(2*np.pi*b*(x+c))/np.cosh((x-f)/e) + d)

#----- SingleTrace class -----
class SingleTrace:
	def __init__(self, file_name, header_lines):
		if type(file_name) == np.ndarray:
			self.constructor_data(file_name, header_lines)
		else:
			self.constructor_file(file_name, header_lines)
	
	def constructor_file(self, file_name, header_lines):
		data = np.genfromtxt(file_name, skip_header=header_lines)
		self.name = file_name
		self.time = data[:, 0]
		self.signal_real = data[:, 1]
		self.signal_imag = data[:, 2]
		self.stage_pos = data[:, 3]
	
	def constructor_data(self, data, name):
		self.name = name
		self.time = data[:, 0]
		self.signal_real = data[:, 1]
		self.signal_imag = data[:, 2]
		self.stage_pos = data[:, 3]
	
	def __add__(self, y):
		n_points = self.time.size
		if n_points != y.time.size:
			raise RuntimeError('The traces cannot be added')
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = self.signal_real + y.signal_real
		out_data[:, 2] = self.signal_imag + y.signal_imag
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_add')
		return out_trace
	
	def __sub__(self, y):
		n_points = self.time.size
		if n_points != y.time.size:
			raise RuntimeError('The traces cannot be subtracted')
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = self.signal_real - y.signal_real
		out_data[:, 2] = self.signal_imag - y.signal_imag
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_sub')
		return out_trace
	
	def abs(self, y):
		self.signal_real = np.abs(self.signal_real)
		self.signal_imag = np.abs(self.signal_imag)
	
	def normalize(self, y):
		out_data = np.zeros((self.time.size, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = self.signal_real/y
		out_data[:, 2] = self.signal_imag/y
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_rsc_{:.2f}'.format(y))
		return out_trace
	
	def shift_time(self, t_offset):
		self.time += t_offset
	
	#Quick method for plotting real and imaginary parts of a single trace
	def plot(self, type='t'):
		font_size = 18
		if type == 't':
			x_label = 't (ps)'
			x_to_plot = self.time
		elif type == 'p':
			x_label = 'Stage pos. (mm)'
			x_to_plot = self.stage_pos
		else:
			raise RuntimeError('Type not defined')
		fig1 = plt.figure(figsize=figsize_single)
		ax1 = fig1.add_subplot(1,2,1)
		ax1.plot(x_to_plot, self.signal_real, '-ok', markersize=3)
		ax1.set_xlabel(x_label, fontsize=font_size)
		ax1.set_ylabel('Real signal', fontsize=font_size)
		ax1.tick_params(axis='both', labelsize=font_size)
		ax1.grid(True)
		ax2 = fig1.add_subplot(1,2,2)
		ax2.plot(x_to_plot, self.signal_imag, '-or', markersize=3)
		ax2.set_xlabel(x_label, fontsize=font_size)
		ax2.set_ylabel('Imaginary signal', fontsize=font_size)
		ax2.tick_params(axis='both', labelsize=font_size)
		ax2.grid(True)
		# fig1.suptitle('Trace n. {:d}'.format(self.index), fontsize=font_size+2)
		fig1.tight_layout()
		plt.show()
		# return fig1
	
	#Fitting method - type[0] contains the signal ('r' or 'i') and type[1] contains the function ('e' or 'c') 
	def fit_damping(self, type, p0_list, fix_relaxation=False):
		if type[0] == 'r':
			y_to_fit = self.signal_real
		elif type[0] == 'i':
			y_to_fit = self.signal_imag
		else:
			raise(RuntimeError('Fit type not defined'))
			
		if type[1] == 'e':
			fit_function = fit_expFunc
		elif type[1] == 'c':
			if len(p0_list) == 6:
				fit_function = fit_coshFunc
			elif len(p0_list) == 5:
				fit_function = lambda x, *p : fit_coshFunc(x, p[0], p[1], p[2], p[3], p[4], 0)
			else:
				raise(RuntimeError('Wrong number of p0 parameters'))
		else:
			raise(RuntimeError('Fit type not defined'))
			
		if type[1] == 'c' and len(p0_list) == 6 and fix_relaxation:
			fit_function = lambda x, *p : fit_coshFunc(x, p[0], p[1], p[2], p[3], p0_list[4], p[4])
			del p0_list[4]
		
		popt, pcov = curve_fit(fit_function, self.time, y_to_fit, p0=tuple(p0_list), maxfev=400*(len(p0_list)+1)) #maxfev=200*(len(p0_list)+1) is the default
		perr = np.sqrt(np.diag(pcov))
		fit_res = fit_function(self.time, *popt)
		
		return fit_res, popt, perr
	
	def fft(self, type='r'):
		n_points = self.time.size
		delta_t = (self.time[-1] - self.time[0])/(n_points - 1)
		fs = 1/delta_t
		f = fs*np.arange(0, n_points, 1)/n_points
		if type == 'r':
			data_for_fft = self.signal_real
		elif type == 'i':
			data_for_fft = self.signal_imag
		else:
			raise(RuntimeError('Data type not defined'))
		fft_res = np.fft.fft(data_for_fft)
		fft_amplitude = np.abs(fft_res)/n_points
		fft_phase = np.angle(fft_res)
		
		return f, fft_amplitude, fft_phase
	
	def export(self, out_name):
		data_to_save = np.zeros((self.time.size, 4), dtype=np.float_)
		data_to_save[:, 0] = self.time
		data_to_save[:, 1] = self.signal_real
		data_to_save[:, 2] = self.signal_imag
		data_to_save[:, 3] = self.stage_pos
		np.savetxt(out_name, data_to_save, fmt='%.6e', delimiter='\t')
		print('Exported file for ' + self.name)
#-----------------------------

#Open multiple data files
def open_multiple(input_folder, base_name, extension, header_lines=7):
	file_list = os.listdir(input_folder)
	traces_list = []
	for f in file_list:
		file_name, file_extension = os.path.splitext(f)
		if file_name.startswith(base_name) and file_extension == extension:
			print(f)
			traces_list.append(SingleTrace(input_folder+f, header_lines))
	return traces_list

#Average the traces contained in traces_list
def average_traces(traces_list):
	n_traces = len(traces_list)
	n_points = traces_list[0].time.shape[0]
	out_data = np.zeros((n_points, 4), dtype=np.float_)
	for i in range(n_traces):
		if traces_list[i].time.shape[0] != n_points:
			raise(RuntimeError('Wrong number of points at trace {:d}'.format(i)))
		else:
			out_data[:, 0] += traces_list[i].time
			out_data[:, 1] += traces_list[i].signal_real
			out_data[:, 2] += traces_list[i].signal_imag
			out_data[:, 3] += traces_list[i].stage_pos
	out_trace = SingleTrace(out_data/n_traces, 'avg_{:d}'.format(n_traces))
	return out_trace

#Crop the "in_trace" trace	
def crop_trace(in_trace, limits=(0,)):
	if len(limits) == 1:
		filter = in_trace.time >= limits[0]
	else:
		filter = np.logical_and(data_x >= limits[0], data_x < limits[1])
	n_points = in_trace.time[filter].size
	out_data = np.zeros((n_points, 4), dtype=np.float_)
	out_data[:, 0] = in_trace.time[filter]
	out_data[:, 1] = in_trace.signal_real[filter]
	out_data[:, 2] = in_trace.signal_imag[filter]
	out_data[:, 3] = in_trace.stage_pos[filter]
	out_trace = SingleTrace(out_data, in_trace.name+'_crop')
	return out_trace

#Get the selected file's path in a string
def get_path(start_path='', filter=(('Data file', '*.dat'),('All files', '*'))):
	root = tk.Tk() #hide the root window
	root.withdraw() #hide the root window
	return filedialog.askopenfilename(initialdir=start_path, title='Select input file', filetypes=filter)