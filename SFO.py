# _summary_
# ==================================================
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from Functions import Get_Functions_details
# ==================================================
__all__ = ['SFO']


class SFO(object):

    def __init__(self, fobj: callable,
                 NSF, NSA, # number of sailfish and sardine
                 lb, ub, # lower and upper bounds
                 ndim,
                 max_iter=500,
                 ):
               
        self.F = fobj
        self.NSF = NSF
        self.NSA = NSA
        self.lb = lb
        self.ub = ub
        self.ndim = ndim
        self.max_iter = max_iter
        
        self.init_SFpos = np.random.uniform(lb, ub, (NSF, ndim))
        self.init_SApos = np.random.uniform(lb, ub, (NSA, ndim))
        self.SFpos = self.init_SFpos.copy()
        self.SApos = self.init_SApos.copy()
        self.SFfit = np.apply_along_axis(self.F, 1, self.SFpos)
        self.SAfit = np.apply_along_axis(self.F, 1, self.SApos)
                
        self.elite_SFidx = np.argmin(self.SFfit)
        self.elite_SFfit = self.SFfit[self.elite_SFidx]
        self.elite_SFpos = self.SFpos[self.elite_SFidx]
        
        self.injured_SAidx = np.argmin(self.SAfit)
        self.injured_SAfit = self.SAfit[self.injured_SAidx]
        self.injured_SApos = self.SApos[self.injured_SAidx]
        
        # parameters for SFO
        self.A = 4.
        self.epsilon = 0.001
        
    def run(self):
        best_fit_rec = []
        best_pos = self.elite_SFpos.copy()
        best_fit = self.elite_SFfit.copy()
        best_fit_rec.append(self.SFfit[self.elite_SFidx])
        
        for i in range(self.max_iter):
            self.SF_pos_update()
            AP = self.calc_AP(i)
            if AP < 0.5:
                # Update a subset of sardine positions
                self.SA_partial_update(AP)
            else:
                # Update all sardine positions
                self.SA_allpos_update(AP)       
            
            self.SFfit = np.apply_along_axis(self.F, 1, self.SFpos)
            self.SAfit = np.apply_along_axis(self.F, 1, self.SApos)
    
            if np.min(self.SAfit) < best_fit:

                self.SFpos[self.elite_SFidx] = self.SApos[self.injured_SAidx].copy()
                self.SFfit[self.elite_SFidx] = self.SAfit[self.injured_SAidx].copy()
                
                best_fit_rec.append(best_fit)
                best_fit = np.min(self.SAfit)
                best_pos = self.SApos[np.argmin(self.SAfit)].copy()
                
                self.SApos = np.delete(self.SApos, self.injured_SAidx, axis=0)
                self.SAfit = np.delete(self.SAfit, self.injured_SAidx)
                self.NSA -= 1
                
                self.elite_SFidx   = np.argmin(self.SFfit)
                self.elite_SFfit   = self.SFfit[self.elite_SFidx]
                self.elite_SFpos   = self.SFpos[self.elite_SFidx]
                self.injured_SAidx = np.argmin(self.SAfit)
                self.injured_SAfit = self.SAfit[self.injured_SAidx]
                self.injured_SApos = self.SApos[self.injured_SAidx]
            
            else:
                best_fit_rec.append(best_fit)
        
        return best_pos, best_fit, best_fit_rec
    
    def calc_AP(self, Itr):
        return self.A * (1 - (2 * Itr * self.epsilon))

    def calc_PD(self):
        return 1 - (self.NSF / (self.NSF + self.NSA))

    def calc_lambda(self):
        return 2 * np.random.rand(1) * self.calc_PD() - self.calc_PD()
    
    def SF_pos_update(self):
        """
        Update sardine positions based on the elite sardine position.
        """
        self.SFpos = self.elite_SFpos[np.newaxis, :] - \
            self.calc_lambda() * \
            (np.random.rand(self.NSF)[:, np.newaxis] * (self.elite_SFpos + self.injured_SApos)/2 - self.SFpos)

    def SA_allpos_update(self, AP):
        """
        Update all sardine positions based on Attack Power.
        """
        self.SApos = np.random.rand(self.NSA)[:, np.newaxis] * \
            (self.elite_SFpos[np.newaxis, :] - self.SApos + AP)
            
    def SA_partial_update(self, AP):
        """
        Update a fraction of sardine positions based on Attack Power.
        """
        num_to_update = int(self.NSA * AP)
        num_of_var    = int(self.ndim * AP)
        
        select_SAidx  = np.random.choice(self.NSA, num_to_update, replace=False)
        select_varidx = np.random.choice(self.ndim, num_of_var, replace=False)
        
        self.SApos[select_SAidx, :] = np.random.rand(num_to_update)[:, np.newaxis] * \
            (self.elite_SFpos[np.newaxis, :] - self.SApos[select_SAidx, :] + AP)


# ==================================================

def main():
    Function_name = "F10"
    fobj, lb, ub, ndim = Get_Functions_details(Function_name)
    NSF = 3
    NSA = 100
    
    
    sfo = SFO(fobj, NSF, NSA, lb, ub, ndim)
    
    sfo.run()
    
    pass

# ==================================================

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))