import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

import numpy as np
import sys
sys.path.insert(0, "FIt-SNE/") 
from fast_tsne import fast_tsne


class TreeSNE:

    def __init__(self,
                 data,
                 map_dims,
                 final_df,
                 levels):
        
        self.data = data
        self.map_dims = map_dims
        self.levels = levels
        self.perplexity = data.shape[0] ** 0.5
        self.r = np.exp(np.log(final_df)/levels)
        self.fits = np.zeros((levels+1, data.shape[0], map_dims))

    
    def fit(self):

        self.fits[0] = fast_tsne(self.data, 
                    map_dims = self.map_dims, 
                    perplexity = self.perplexity, 
                    search_k = 150*int(self.perplexity), 
                    df = self.r**0, 
                    late_exag_coeff=12
                    )
        
        for i in range(1, self.levels + 1):
            print(i)
            self.fits[i] = fast_tsne(self.data, 
                                map_dims = self.map_dims, 
                                perplexity = self.perplexity**(self.r**i), 
                                df = self.r**i, 
                                initialization=self.fits[i-1], 
                                late_exag_coeff=12,
                                search_k = 150 * int(self.perplexity**(self.r**i))
                                )
            
    def visualize(self,
                  fits = None,
                  save = None):
        
        if type(fits) == None:
            X = self.fits
        
        else:
            X = fits

        stack_fits = np.zeros((self.levels + 1, self.data.shape[0], self.map_dims+1))
        for i in range(self.levels+1):
            stack_fits[i] = np.column_stack((X[i], i * np.ones(self.data.shape[0])))


        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')


        for i in range(self.levels):
            ax.scatter(xs = stack_fits[i][:, 0], 
                       ys=stack_fits[i][:, 1],  
                       zs=stack_fits[i][:, 2], 
                       c='black', s=4, alpha=300/self.data.shape[0])

        if save != None:
            plt.savefig(save)
