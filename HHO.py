# _summary_
# ==================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from time import perf_counter
# ==================================================
__all__ = ['HHO']

class HHO(object):
    """"""
    def __init__(self, fobj: callable, 
                 Nhawks: int, 
                 lb: float, ub: float, 
                 ndim: int, 
                 max_iter: int):
        
        self.F = fobj
        self.Nhawks = Nhawks
        self.lb = lb
        self.ub = ub
        self.ndim = ndim
        self.max_iter = max_iter

        self.rabbit_loc = np.zeros(ndim)
        self.rabbit_energy = float("inf")
        
        self.X = self.initialization(Nhawks, ndim, ub, lb) # Harris’ hawks init
        self.CNVG = np.zeros(max_iter)
    
    def update(self, t):
        self.X = np.clip(self.X, self.lb, self.ub)
            
        fitness = np.apply_along_axis(self.F, 1, self.X)
        
        # update rabbit location and energy
        min_index = np.argmin(fitness)
        if fitness[min_index] < self.rabbit_energy:
            self.rabbit_energy = fitness[min_index]
            self.rabbit_loc = self.X[min_index, :].copy()
        
        E1 = 2 * (1 - t / self.max_iter)  # energy decay
        
        # 計算逃逸能量和更新位置
        E0 = 2 * np.random.rand(self.Nhawks) - 1  # -1 < E0 < 1
        Escaping_Energy = E1 * E0
        
        # Classification -> exploration / exploitation phases
        abs_Energy = np.abs(Escaping_Energy)
        exploration_mask = abs_Energy >= 1 
        
        # Exploration phase
        if np.any(exploration_mask):
            # Eq.1

            q = np.random.rand(self.Nhawks)
            rand_Hawk_indices = np.random.randint(0, self.Nhawks, self.Nhawks)
            X_rand = self.X[rand_Hawk_indices]
            
            self.X[exploration_mask & (q < 0.5)] = (
                X_rand[exploration_mask & (q < 0.5)] - 
                np.random.rand() * 
                np.abs(
                    X_rand[exploration_mask & (q < 0.5)] -
                    2 * np.random.rand() * self.X[exploration_mask & (q < 0.5)]
                )
            )
            self.X[exploration_mask & (q >= 0.5)] = (
                self.rabbit_loc - np.mean(self.X, axis=0) -
                np.random.rand() *
                ((self.ub - self.lb) * np.random.rand() + self.lb)
            )
        
        # 利用階段
        if np.any(~exploration_mask):
            r = np.random.rand(self.Nhawks)
            Jump_strength = 2 * (1 - np.random.rand(self.Nhawks))
            
            for i in range(self.Nhawks):
                if ~exploration_mask[i]:
                    if r[i] >= 0.5 and abs_Energy[i] < 0.5:
                        self.X[i, :] = self.rabbit_loc - Escaping_Energy[i] * np.abs(self.rabbit_loc - self.X[i, :])
                    elif r[i] >= 0.5 and abs_Energy[i] >= 0.5:
                        self.X[i, :] = (
                            self.rabbit_loc - Escaping_Energy[i] *
                            np.abs(Jump_strength[i] * self.rabbit_loc - self.X[i, :])   
                        )
                    else:
                        X1 = (
                            self.rabbit_loc - Escaping_Energy[i] *
                            np.abs(Jump_strength[i] * self.rabbit_loc - self.X[i, :])
                        )
                        if self.F(X1) < self.F(self.X[i, :]):
                            self.X[i, :] = X1
                        else:
                            X2 = (
                                self.rabbit_loc - Escaping_Energy[i] *
                                np.abs(Jump_strength[i] * self.rabbit_loc - np.mean(self.X, axis=0)) +
                                self.Levy(self.ndim)
                            )
                            if self.F(X2) < self.F(self.X[i, :]):
                                self.X[i, :] = X2
        
    def run(self):
        for t in range(self.max_iter):
            self.update(t)
            
            # best energy at each iteration
            self.CNVG[t] = self.rabbit_energy
        
        return self.rabbit_energy, self.rabbit_loc, self.CNVG
        
    @ staticmethod
    def initialization(N, ndim, ub, lb):
        if np.isscalar(ub):
            X = np.random.rand(N, ndim) * (ub - lb) + lb
        else:
            X = np.zeros((N, ndim))
            for i in range(ndim):
                X[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]
        return X
    
    @ staticmethod
    def Levy(ndim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(ndim) * sigma
        v = np.random.randn(ndim)
        
        return u / np.abs(v)**(1 / beta)
    
    def Visualization(self, save_path=os.path.join(os.getcwd(), 'HHO_Visualization.mp4')):
        if self.ndim != 2:
            raise ValueError("Visualization is only available for 2-dimensional problems")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(self.lb, self.ub, 100)
        y = np.linspace(self.lb, self.ub, 100)
        xx, yy = np.meshgrid(x, y)

        z = np.apply_along_axis(self.F, 1, np.vstack((xx.flatten(), yy.flatten())).T).reshape(100, 100)
        
        ax.plot_surface(xx, yy, z.reshape(100, 100), cmap='coolwarm', alpha=.5)
        Rabbitpos, = ax.plot(self.rabbit_loc[0], self.rabbit_loc[1], self.rabbit_energy, 'yo', ms=5)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 10)
        
        def update_plot(t):
            print(f"\rrendering frame {t+1}/{self.max_iter}", end='')
            self.update(t)
            
            Rabbitpos.set_data(self.rabbit_loc[0], self.rabbit_loc[1])
            Rabbitpos.set_3d_properties(self.rabbit_energy)
            return Rabbitpos,
            
        writer = FFMpegWriter(fps=10)
        anim = FuncAnimation(fig, update_plot, frames=self.max_iter)
        anim.save(save_path, writer=writer)
        
    
# ==================================================

def main():
    pass

# ==================================================

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))


