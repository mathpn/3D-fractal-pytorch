import numpy as np
from scipy.stats import linregress
from torch.nn import AvgPool3d
import torch

### The Fractal class takes as input a 3D numpy array and a threshold
### The threshold is used to binarize the array
### Values > threshold are included in the binarized fractal representation

class Fractal:

    def __init__(self, Z, threshold = 1):
        # Set the default tensor as CUDA Tensor
        # If you don't have a CUDA GPU, remove all 'cuda' from the lines below
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Make the array binary
        self.Z = np.array((Z > threshold), dtype = int)

        # Get the list of sizes for box-counting
        self.sizes = self.get_sizes()

        # Perform box-counting
        self.count, self.lac = self.get_count()

        # Get fractal dimensionality
        slope, _, R, _, self.st_er = linregress(np.log(self.sizes), np.log(self.count))
        self.Db = -slope
        self.Rsquared = R**2

        # Lacunarity measures
        self.mean_lac = np.mean(self.lac)
        # 1 is added to avoid log of 0
        self.lac_reg_coeff, _, R_lac, _, self.st_er_lac = linregress(np.log(self.sizes), np.log(self.lac + 1))
        self.Rsquared_lac = R_lac**2
        
        return None

    def get_sizes(self):
        # Minimal dimension of image
        p = min(self.Z.shape)

        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p/2)/np.log(2))

        # Extract the exponent
        n = int(np.log(n)/np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)
        return sizes

    def get_count(self):
        # Pre-allocation
        counts = np.empty((len(self.sizes)))
        lacs = np.empty((len(self.sizes)))
        index = 0

        # Transfer the array to a 4D CUDA Torch Tensor
        temp = torch.Tensor(self.Z).unsqueeze(0)

        # Box-counting
        for size in self.sizes:
            # i is a variable to perform box-counting at multiple orientations
            i = 0
            count_u = 0
            lac = 0

            while i in range(4) and ((i*(size/4) + size) < min(self.Z.shape)-1):
                temp = temp[:, i:, i:, i:]
                stride = (int(size/2), int(size/2), int(size/2))
                pool = AvgPool3d(kernel_size = (size, size, size), stride = stride, padding = int(size/2))
                # Performs optimized 3D average pooling for box-counting
                S = pool(temp)
                # If S == 1, then there is no edge on that box
                count = torch.sum(torch.where((S > 0) & (S < 1), torch.tensor([1]), torch.tensor([0]))).item()

                # Add to box counting
                count_u += count

                # Calculate Lacunarity
                u = torch.mean(S).item()
                sd = torch.std(S, unbiased = False).item()

                # 0.1 is added to avoid possible error due to division by 0
                lac += (sd/(u+0.1))**2
                i += 1

            # Avoid division by 0
            if i != 0:
                count_u *= 1/i
                lac *= 1/i

            # Results are given as an average for all orientations
            counts[index] = count_u
            lacs[index] = lac
            index += 1

        return counts, lacs