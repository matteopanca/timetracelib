import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from copy import deepcopy

from timetrace import figsize_double
from timetrace import figsize_single

#----- Functions -----

# LLG eq. in cartesian components
def llg_eq(y, t, c_1, c_2, H, D):
	H_eff = H - D*y
	prod_vett = np.zeros(3, dtype=np.float_)
	prod_vett[0] = y[1]*H_eff[2] - H_eff[1]*y[2]
	prod_vett[1] = y[2]*H_eff[0] - H_eff[2]*y[0]
	prod_vett[2] = y[0]*H_eff[1] - H_eff[0]*y[1]
	
	dy = np.zeros(3, dtype=np.float_)
	dy[0] = -c_1*prod_vett[0] - c_2*(y[1]*prod_vett[2] - prod_vett[1]*y[2])
	dy[1] = -c_1*prod_vett[1] - c_2*(y[2]*prod_vett[0] - prod_vett[2]*y[0])
	dy[2] = -c_1*prod_vett[2] - c_2*(y[0]*prod_vett[1] - prod_vett[0]*y[1])
	return dy

# Fit functions
def fit_expFunc(x, a, b, c, d, e):
	return (np.exp(-x/e)*a*np.cos(2*np.pi*b*(x+c)) + d)

def fit_coshFunc(x, a, b, c, d, e, f):
	return (a*np.cos(2*np.pi*b*(x+c))/np.cosh((x-f)/e) + d)

#----- LLG class -----
class LLG:
	def __init__(self):
		self.gamma = 2.211e5 #in (rad/s)*(m/A)
		self.alpha = 0 #"alpha = 0" means no damping
		self.Ms = 0 #in A/m
		self.demag_factors = np.zeros(3, dtype=np.float_)
		self.H = np.zeros(3, dtype=np.float_) #in A/m
		self.time = None
		self.m = None
		self.delta_m = None
	
	def copy(self):
		return deepcopy(self)
	
	def set_alpha(self, alpha):
		self.alpha = alpha
	
	def set_Ms(self, Ms):
		self.Ms = Ms
	
	def set_demagFactors(self, demag_factors):
		for i in range(3):
			self.demag_factors[i] = demag_factors[i]
	
	def set_h(self, H, unit='mT'):
		if unit == 'Oe':
			h_conv = 1e3/(4*np.pi) #from Oe to A/m
		elif unit == 'mT':
			h_conv = 1e4/(4*np.pi) #from mT to A/m
		else:
			raise RuntimeError('Unit not yet supported')
		for i in range(3):
			self.H[i] = h_conv*H[i]
	
	def solve(self, time_step, time_end, y0):
		self.time = np.arange(0, time_end+time_step, time_step, dtype=np.float_)
		c_1 = self.gamma/(1 + self.alpha**2)
		c_2 = self.gamma*self.alpha/(1 + self.alpha**2)
		self.m = odeint(llg_eq, y0, self.time, args=(c_1,c_2,self.H,self.Ms*self.demag_factors))
		self.delta_m = np.copy(self.m)
		for i in range(3):
			self.delta_m[:, i] -= self.delta_m[-1, i]
	
	def fft(self, consider_delta=False):
		n_points = self.time.size
		delta_t = (self.time[-1] - self.time[0])/(n_points - 1)
		fs = 1/delta_t
		f = fs*np.arange(0, n_points, 1)/n_points
		
		if consider_delta:
			data = self.delta_m
		else:
			data = self.m
		
		fft_amplitude = np.zeros((n_points, 3), dtype=np.float)
		fft_phase = np.zeros((n_points, 3), dtype=np.float)
		for i in range(3):
			fft_res = np.fft.fft(data[:, i])
			fft_amplitude[:, i] = np.abs(fft_res)/n_points
			fft_phase[:, i] = np.angle(fft_res)
		
		return f, fft_amplitude, fft_phase
	
	def fit_damping(self, comp, est_freq):
		y_to_fit = self.m[:, comp]
		est_amp = (np.amax(y_to_fit) - np.amin(y_to_fit))/2.
		est_off = np.mean(y_to_fit)
		est_p0 = (est_amp, est_freq, 0., est_off, 1./est_freq) #this has to be a tuple of 5 elements
		popt, pcov = curve_fit(fit_expFunc, self.time, y_to_fit, p0=est_p0, maxfev=400*(len(est_p0)+1)) #maxfev=200*(len(p0_list)+1) is the default
		perr = np.sqrt(np.diag(pcov))
		fit_res = fit_expFunc(self.time, *popt)
		return fit_res, popt, perr
	
	def plot(self, size=figsize_single):
		my_color = ['b', 'r', 'g']
		font_size = 18
		fig1 = plt.figure(figsize=size)
		ax1 = fig1.add_subplot(1,1,1)
		for i in range(3):
			ax1.plot(self.time, self.m[:, i], '-o', color=my_color[i], markersize=3, label='Comp. {:d}'.format(i))
		ax1.set_xlim([self.time[0], self.time[-1]])
		ax1.set_xlabel('t (s)', fontsize=font_size)
		ax1.set_ylabel('Norm. magn.', fontsize=font_size)
		ax1.tick_params(axis='both', labelsize=font_size)
		ax1.legend(loc='best', fontsize=18)
		ax1.grid(True)
		fig1.tight_layout()
		plt.show()
		# return fig1