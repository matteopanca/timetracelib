import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from timetrace import figsize_double
from timetrace import figsize_single
# from timetrace import get_path

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
		data = np.genfromtxt(file_name, dtype=np.float_, skip_header=header_lines)
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
	
	def __pow__(self, y):
		n_points = self.time.size
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = self.signal_real**y
		out_data[:, 2] = self.signal_imag**y
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_pow')
		return out_trace
	
	def len(self):
		return self.time.size
	
	def sqrt(self):
		n_points = self.time.size
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = np.sqrt(self.signal_real)
		out_data[:, 2] = np.sqrt(self.signal_imag)
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_sqrt')
		return out_trace
	
	def abs(self):
		n_points = self.time.size
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = np.abs(self.signal_real)
		out_data[:, 2] = np.abs(self.signal_imag)
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_abs')
		return out_trace
	
	def mean(self, type='r'):
		if type == 'r':
			out_value = np.mean(self.signal_real)
		elif type == 'i':
			out_value = np.mean(self.signal_imag)
		else:
			raise(RuntimeError('Type not defined'))
		return out_value
	
	def max(self, type='r'):
		if type == 'r':
			out_value = np.amax(self.signal_real)
		elif type == 'i':
			out_value = np.amax(self.signal_imag)
		else:
			raise(RuntimeError('Type not defined'))
		return out_value
	
	def min(self, type='r'):
		if type == 'r':
			out_value = np.amin(self.signal_real)
		elif type == 'i':
			out_value = np.amin(self.signal_imag)
		else:
			raise(RuntimeError('Type not defined'))
		return out_value
	
	def normalize(self, y):
		out_data = np.zeros((self.time.size, 4), dtype=np.float_)
		out_data[:, 0] = self.time
		out_data[:, 1] = self.signal_real/y
		out_data[:, 2] = self.signal_imag/y
		out_data[:, 3] = self.stage_pos
		out_trace = SingleTrace(out_data, self.name+'_norm_{:.2f}'.format(y))
		return out_trace
	
	def shift_time(self, t_offset):
		self.time += t_offset
		return self
	
	def shift_trace(self, y_offset, type='r'):
		if type == 'r':
			self.signal_real += y_offset
		elif type == 'i':
			self.signal_imag += y_offset
		else:
			raise(RuntimeError('Type not defined'))
		return self
	
	def zero_trace(self, limits, type='r'):
		filter = np.logical_and(self.time >= limits[0], self.time <= limits[1])
		if type == 'r':
			value_to_shift = -np.mean(self.signal_real[filter])
		elif type == 'i':
			value_to_shift = -np.mean(self.signal_imag[filter])
		else:
			raise(RuntimeError('Type not defined'))
		return self.shift_trace(value_to_shift, type)
	
	def crop(self, limits=(0,), type='t'):
		if type == 't':
			to_filter = self.time
		elif type == 's':
			to_filter = self.stage_pos
		else:
			raise(RuntimeError('Type not defined'))
		if len(limits) == 1:
			filter = to_filter >= limits[0]
		else:
			filter = np.logical_and(to_filter >= limits[0], to_filter <= limits[1])
		n_points = self.time[filter].size
		out_data = np.zeros((n_points, 4), dtype=np.float_)
		out_data[:, 0] = self.time[filter]
		out_data[:, 1] = self.signal_real[filter]
		out_data[:, 2] = self.signal_imag[filter]
		out_data[:, 3] = self.stage_pos[filter]
		out_trace = SingleTrace(out_data, self.name+'_crop')
		return out_trace
	
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
	
	#Quick method for plotting real or imaginary parts of a single trace
	def plot(self, type='tr', color='k', size=figsize_single):
		font_size = 18
		if type[0] == 't':
			x_label = 't (ps)'
			x_to_plot = self.time
		elif type[0] == 'p':
			x_label = 'Stage pos. (mm)'
			x_to_plot = self.stage_pos
		else:
			raise RuntimeError('Type not defined')
		if type[1] == 'r':
			y_label = 'X channel'
			y_to_plot = self.signal_real
		elif type[1] == 'i':
			y_label = 'Y channel'
			y_to_plot = self.signal_imag
		else:
			raise RuntimeError('Type not defined')
		fig1 = plt.figure(figsize=size)
		ax1 = fig1.add_subplot(1,1,1)
		ax1.plot(x_to_plot, y_to_plot, '-o', color=color, markersize=3)
		ax1.set_xlabel(x_label, fontsize=font_size)
		ax1.set_ylabel(y_label, fontsize=font_size)
		ax1.tick_params(axis='both', labelsize=font_size)
		ax1.grid(True)
		fig1.tight_layout()
		plt.show()
		# return fig1
	
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
def open_multiple(input_folder, base_name, extension, header_lines=4):
	file_list = os.listdir(input_folder)
	traces_list = []
	for f in file_list:
		file_name, file_extension = os.path.splitext(f)
		if file_name.startswith(base_name) and file_extension == extension:
			# print(f)
			traces_list.append(SingleTrace(input_folder+f, header_lines))
	return traces_list

#Average the traces contained in traces_list
def average_traces(traces_list):
	n_traces = len(traces_list)
	n_points = traces_list[0].time.size
	out_data = np.zeros((n_points, 4), dtype=np.float_)
	for i in range(n_traces):
		if traces_list[i].time.size != n_points:
			raise(RuntimeError('Wrong number of points at trace {:d}'.format(i)))
		else:
			out_data[:, 0] += traces_list[i].time
			out_data[:, 1] += traces_list[i].signal_real
			out_data[:, 2] += traces_list[i].signal_imag
			out_data[:, 3] += traces_list[i].stage_pos
	out_trace = SingleTrace(out_data/n_traces, 'avg_{:d}'.format(n_traces))
	return out_trace