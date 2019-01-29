import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

from timetrace import figsize_double
from timetrace import figsize_single
# from timetrace import get_path

#----- Functions -----
def fit_erfFunc(x, a, b, c, d, e, f):
	return (a*erf((x - b)/(c*np.sqrt(2))) + d + e*x + f*x**2)

#----- HystLoop class -----
class HystLoop:
	def __init__(self, data, name, units):
		if data.shape[1] == 4:
			index_a0 = 0
			index_a1 = 1
			index_b0 = 2
			index_b1 = 3
			n_points = data.shape[0]
		else:
			index_a0 = 0
			index_a1 = 1
			index_b0 = 0
			index_b1 = 1
			if data.shape[0]%2 == 0:
				n_points = int(data.shape[0]/2)
			else:
				n_points = int(np.ceil(data.shape[0]/2))
		self.field = np.zeros((n_points, 2), dtype=np.float_)
		self.magn = np.zeros((n_points, 2), dtype=np.float_)
		self.direction = np.zeros(2, dtype=np.int_)
		self.field[:, 0] = data[:n_points, index_a0] #first branch
		self.magn[:, 0] = data[:n_points, index_a1] #first branch
		self.field[:, 1] = data[-n_points:, index_b0] #second branch
		self.magn[:, 1] = data[-n_points:, index_b1] #second branch
		self.name = name
		self.units = units
	
	def rescale_field(self, p, unit=None):
		new_field = np.zeros(self.field.shape, dtype=np.float_)
		for i in range(len(p)):
			new_field += p[i]*self.field**i
		self.field = new_field
		if unit is not None:
			self.unit[0] = unit
		return self
	
	def mean(self, n_points=-1):
		if type(n_points) == tuple:
			filter = np.logical_or(self.field < n_points[0], self.field > n_points[1])
			out_value = np.mean(self.magn[filter])
		else:
			if n_points == -1:
				out_value = np.mean(self.magn)
			else:
				if n_points > self.field.shape[0]:
					raise RuntimeError('Too many points for the mean value')
				else:
					data_to_consider = np.zeros((n_points, 4), dtype=np.float_)
					data_to_consider[:, 0:2] = self.magn[:n_points, 0:2]
					data_to_consider[:, 2:4] = self.magn[-n_points:, 0:2]
					out_value = np.mean(data_to_consider)
		return out_value
	
	def amp(self, n_points):
		if type(n_points) == tuple:
			filter_0 = self.field < n_points[0]
			filter_1 = self.field > n_points[1]
			out_value = np.abs(np.mean(self.magn[filter_0]) - np.mean(self.magn[filter_1]))/2
		else:
			if n_points > self.field.shape[0]:
				raise RuntimeError('Too many points for the mean value')
			else:
				data_to_consider = np.zeros((n_points, 4), dtype=np.float_)
				data_to_consider[:, 0] = self.magn[:n_points, 0]
				data_to_consider[:, 1] = self.magn[-n_points:, 1]
				data_to_consider[:, 2] = self.magn[-n_points:, 0]
				data_to_consider[:, 3] = self.magn[:n_points, 1]
				out_value = np.abs(np.mean(data_to_consider[:, 0:2]) - np.mean(data_to_consider[:, 2:4]))/2
		return out_value
	
	def center(self, n_points=-1):
		mean_value = self.mean(n_points)
		out_data = np.zeros((self.field.shape[0], 4), dtype=np.float_)
		out_data[:, 0] = self.field[:, 0]
		out_data[:, 1] = self.magn[:, 0] - mean_value
		out_data[:, 2] = self.field[:, 1]
		out_data[:, 3] = self.magn[:, 1] - mean_value
		out_loop = HystLoop(out_data, self.name+'_center_{:d}'.format(n_points), self.units)
		return out_loop
	
	def normalize(self, n_points, my_value=None):
		mean_value = self.mean(n_points)
		if my_value is not None:
			amp_value = my_value
		else:
			amp_value = self.amp(n_points)
		out_data = np.zeros((self.field.shape[0], 4), dtype=np.float_)
		out_data[:, 0] = self.field[:, 0]
		out_data[:, 1] = (self.magn[:, 0] - mean_value)/amp_value
		out_data[:, 2] = self.field[:, 1]
		out_data[:, 3] = (self.magn[:, 1] - mean_value)/amp_value
		out_loop = HystLoop(out_data, self.name+'_norm_{:.2f}'.format(amp_value), [self.units[0], 'a.u.'])
		return out_loop
	
	#for a regular field sweep, (delta_points) propto (delta_t)
	def remove_drift(self, n_points=5):
		position_zero = n_points/2 #it does not have to be int...
		delta = (np.mean(self.magn[-n_points:, 1])-np.mean(self.magn[:n_points, 0]))/(2*self.field.shape[0]-n_points-1)
		
		out_data = np.zeros((self.field.shape[0], 4), dtype=np.float_)
		out_data[:, 0] = self.field[:, 0]
		out_data[:, 2] = self.field[:, 1]
		counter = 0
		for i in range(2):
			for j in range(self.field.shape[0]):
				out_data[j, 2*i+1] = self.magn[j, i] - delta*(counter-position_zero)
				counter += 1
		out_loop = HystLoop(out_data, self.name+'_noDrift', self.units)
		return out_loop
	
	def fit_loop(self, n_points=5):
		d_guess = self.mean(n_points)
		e_guess = 0
		f_guess = 0
		
		fit_res = np.zeros(self.field.shape, dtype=np.float_)
		pOpt_out = []
		pErr_out = []
		for i in range(2):
			direction = np.sign((self.magn[-1, i] - self.magn[0, i])*(self.field[-1, i] - self.field[0, i]))
			a_guess = direction*self.amp(n_points)
			b_guess = self.field[np.argmin(np.abs(self.magn[:, i]-d_guess)), i]
			c_guess = np.abs(b_guess)/10 #rough estimate, just for the odg...
			p0_tuple = (a_guess, b_guess, c_guess, d_guess, e_guess, f_guess)
			# print(p0_tuple)
			
			popt, pcov = curve_fit(fit_erfFunc, self.field[:, i], self.magn[:, i], p0=p0_tuple, maxfev=400*(len(p0_tuple)+1)) #maxfev=200*(len(p0_list)+1) is the default
			fit_res[:, i] = fit_erfFunc(self.field[:, i], *popt)
			pOpt_out.append(popt)
			pErr_out.append(np.sqrt(np.diag(pcov)))
		return fit_res, pOpt_out, pErr_out
	
	def plot(self, fit_res=None, size=figsize_single):
		font_size = 18
		fig1 = plt.figure(figsize=size)
		ax1 = fig1.add_subplot(1,1,1)
		ax1.plot(self.field[:, 0], self.magn[:, 0], '--ok', markersize=4, label='Branch 0')
		ax1.plot(self.field[:, 1], self.magn[:, 1], '--or', markersize=4, label='Branch 1')
		if fit_res is not None:
			ax1.plot(self.field[:, 0], fit_res[:, 0], '-g', label='Fit 0')
			ax1.plot(self.field[:, 1], fit_res[:, 1], '-b', label='Fit 1')
		ax1.set_xlabel('Field ({:s})'.format(self.units[0]), fontsize=font_size)
		ax1.set_ylabel('Signal ({:s})'.format(self.units[1]), fontsize=font_size)
		ax1.set_title(self.name, fontsize=font_size+2)
		ax1.tick_params(axis='both', labelsize=font_size)
		ax1.grid(True)
		ax1.legend(loc='best', fontsize=font_size)
		fig1.tight_layout()
		plt.show()
		# return fig1
	
	def export(self, out_name):
		n_points = self.field.shape[0]
		data_to_save = np.zeros((2*n_points, 2), dtype=np.float_)
		data_to_save[:n_points, 0] = self.field[:, 0]
		data_to_save[-n_points:, 0] = self.field[:, 1]
		data_to_save[:n_points, 1] = self.magn[:, 0]
		data_to_save[-n_points:, 1] = self.magn[:, 1]
		np.savetxt(out_name, data_to_save, fmt='%.6e', delimiter='\t')
		print('Exported file for ' + self.name)
#-----------------------------

#Open multiple-loop file
def open_loops(file_name, n_loops=1, units=['A', 'mV']):
	data = np.genfromtxt(file_name, dtype=np.float_)
	if data.shape[0]%n_loops != 0:
		raise RuntimeError('Wrong number of loops')
	else:
		n_points = int(data.shape[0]/n_loops)
	
	if n_loops == 1:
		loops_list = HystLoop(data, file_name.split('\\')[-1], units)
	else:
		loops_list = []
		for i in range(n_loops):
			loops_list.append(HystLoop(data[i*n_points:(i+1)*n_points, :], file_name.split('\\')[-1]+'_{:d}'.format(i)), units)
	return loops_list

#Open multiple files (better if with one loop each)
# def open_multiple(path, file_list, n_loops=1, units=['A', 'mV']):
	# loops_list = []
	# for s in file_list:
		# loops_list.append(open_loops(path+s, n_loops=n_loops, units=units))
	# return loops_list

#Average the loops contained in loops_list
def average_loops(loops_list):
	n_loops = len(loops_list)
	n_points = loops_list[0].field.shape[0]
	out_data = np.zeros((n_points, 4), dtype=np.float_)
	for i in range(n_loops):
		if loops_list[i].field.shape[0] != n_points:
			raise(RuntimeError('Wrong number of points at loop {:d}'.format(i)))
		else:
			out_data[:, 0] += loops_list[i].field[:, 0]
			out_data[:, 1] += loops_list[i].magn[:, 0]
			out_data[:, 2] += loops_list[i].field[:, 1]
			out_data[:, 3] += loops_list[i].magn[:, 1]
	out_loop = HystLoop(out_data/n_loops, 'avg_{:d}'.format(n_loops), loops_list[0].units)
	return out_loop