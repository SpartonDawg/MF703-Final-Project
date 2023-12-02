from scipy.stats import qmc
import numpy as np

def generate_lhs_samples(num_samples, dimensions, max_weight):
    sampler = qmc.LatinHypercube(d=dimensions)
    valid_samples = []

    while len(valid_samples) < num_samples:
        samples = sampler.random(n=num_samples * 10)  # Generate more to ensure we get enough valid ones
        
        scaled_samples = qmc.scale(samples, l_bounds=[0]*dimensions, u_bounds=[max_weight]*dimensions)
        
        for sample in scaled_samples:
            sample_sum = sum(sample)
            if sample_sum <= 1:
                corrected_sample = sample / sample_sum
                if all(weight <= max_weight for weight in corrected_sample):
                    valid_samples.append(corrected_sample)
                    if len(valid_samples) >= num_samples:
                        break

    return np.array(valid_samples)

# Parameters
num_samples = 50000  
dimensions = 8      
max_weight = 0.3     


lhs_samples = generate_lhs_samples(num_samples, dimensions, max_weight)


print(lhs_samples.shape)
