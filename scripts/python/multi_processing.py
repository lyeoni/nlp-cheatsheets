import os
import operator, functools
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

def make_double(n):
	print('PID[{}]:\t{}'.format(os.getpid(), n))
	return n*2

def multi_processing(func, inputs, n_max_workers=None):
	p_executor = ProcessPoolExecutor(max_workers=n_max_workers)

	with p_executor as exec:
		outputs = exec.map(make_double, inputs)
		# Concatenate outputs(Generator) if the dtype of inputs is List
		if type(outputs) == List:
			functools.reduce(operator.iconcat, list(outputs), [])
		else:
			outputs = list(outputs)
	
	return outputs

if __name__=='__main__':
	with open('../../sample/zero_to_hundred.txt') as reader:
		lines = reader.readlines()
		lines = list(map(lambda x: int(x), lines))
	
	outputs = multi_processing(func=make_double, inputs=lines)
	print(len(outputs), outputs)
