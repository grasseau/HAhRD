import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

fname='sq_cells_data/sq_cells_dict_res_473,473_len_0.7.pkl'
fhandle=open(fname)
sq_cells_dict=pickle.load(fhandle)
fhandle.close()


center_x=np.empty((473,473),dtype=np.float64)
center_y=np.empty((473,473),dtype=np.float64)
for id,cell in sq_cells_dict.iteritems():
    i,j=id
    x,y=cell.center.coords[0]
    center_x[i,j]=x
    center_y[i,j]=y

plt.imshow(center_x)
plt.colorbar()
# plt.set_xlim(-160, 160)
# plt.set_ylim(-160, 160)
# plt.set_aspect(1)
plt.show()

plt.imshow(center_y)
plt.colorbar()
# plt.set_xlim(-160, 160)
# plt.set_ylim(-160, 160)
# plt.set_aspect(1)
plt.show()

print sq_cells_dict[(0,0)].polygon.bounds
print sq_cells_dict[(0,1)].polygon.bounds
print sq_cells_dict[(1,0)].polygon.bounds
