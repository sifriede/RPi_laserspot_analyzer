import numpy as np
import pandas as pd

a = np.load("last_img.npz")
d1 = dict(x_x=a['rawdatax_x'], x_y=a['rawdatax_y'], \
         y_x=a['rawdatay_x'], y_y=a['rawdatay_y'])
d2 = dict([(k, pd.Series(v)) for k, v in d1.items()])
df = pd.DataFrame(d2)
df.to_csv("last_img.dat", sep='\t')
