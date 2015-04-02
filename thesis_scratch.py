print [np.median(np.abs((results[:,2]-results[:,3]))),np.median(np.abs((results[:,2]-results[:,7]))),np.median(np.abs((results[:,2]-results[:,4]))), np.sum(results[:,-1])/float(results.shape[0]),np.sum(results[:,-2])/float(results.shape[0])]

print [np.mean(np.abs((results[:,2]-results[:,3]))),np.mean(np.abs((results[:,2]-results[:,7]))),np.mean(np.abs((results[:,2]-results[:,4]))), np.sum(results[:,-1])/float(results.shape[0]),np.sum(results[:,-2])/float(results.shape[0])]

0.26, 0.55
0.069, 0.198
0.128,0.397
0.107, 0.415
0.221, 0.503

incsub.append(incsub[-1][np.random.choice(range(incsub[-1].shape[0]),incsub[-1].shape[0]/2)])

import itertools

def distinfo(arr, thresh=0.1):
	length = arr.shape[0]
	mat = np.array([np.linalg.norm(arr[i]-arr[j]) for i,j in itertools.product(xrange(length),xrange(length))]).reshape(length,length)
	mat[np.diag_indices(length)] = 1
	mins = np.min(mat,axis=0)
	meds = np.median(mat,axis=0)
	meanmin = np.mean(mins)
	meanmed = np.mean(meds)
	clustmin = np.sum(mins>thresh)/float(length)
	return clustmin, meanmin, meanmed, mins, meds, mat

fig, axarr = plt.subplots(1,5)
xmax = np.max([np.max(dat[3]) for dat in info])
ymax = 30
for ind,dat,num,col in zip(range(5),info,(128,64,32,16,8), ('forestgreen','turquoise','royalblue','darkviolet','slategrey')):
 #   plot(xr, gk(dat[3])(xr), label='%d points' % num,color=col)
 	axarr[ind].hist(dat[3],histtype='stepfilled',alpha=.5,normed=1,label='%d points'% num)
 	axarr[ind].set_xlim(0,xmax)
 	axarr[ind].set_ylim(0,ymax)
	axarr[ind].plot((0.1,0.1),(0,ymax),label='Clustering Threshold',color='red')
legend()
title('Minimum distances between points increase with deletion')
ylabel('Probability Density')
xlabel('Distance')

[0.011554444444444431, 0.0080534444444444447, 0.0067927944444444487, 0.999, 0.88200000000000001]



[-0.473671610935,0.640385452186,-0.584067063847,0.646209196195,0.595124054508,0.976674262592,0.985012571981,0.922373274766,0.900492998183,]
