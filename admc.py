from collections import Counter
import numpy as np

def admc(arr, tdist, steps):
	keys = tdist.keys()
	size = len(arr)
	counts = Counter(keys)
	print counts
	totalstates = float(len(keys))

	current = np.random.randint(arr.shape[0])
	cur_val = arr[current]
	cur_lik = tdist[cur_val]/(counts[cur_val]/totalstates)

	states = []
	# start mcmc loop
	for step in xrange(steps):
		new = ((-1,1)[int(np.random.random()<.5)]+current)%size
#		print new
		new_val = arr[new]
		new_lik = tdist[new_val]/(counts[new_val]/totalstates)

		totalstates += 1
		counts[new_val] += 1

		a = new_lik/cur_lik
		if a > np.random.random():
			current = new
			cur_val = new_val
			cur_lik = new_lik
		states.append((cur_val, cur_lik))
	return np.array(states), counts