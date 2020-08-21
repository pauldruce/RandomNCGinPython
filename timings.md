Using type (3,1) as my benchmark, with 10 epochs of chain_length=1000
Going to use timeit for benchmark them. 

Original code: 928.0804097652435 seconds for 20,000
Type specified: 485.5506980419159 seconds for 20,000
With numba+parallel+fast math: 544.120138168335 seconds for 20,000
With numba+fast math, and parallel only sometimes: 4min 8s ± 4.85 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
