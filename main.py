import AstroClass as AC
import numpy as np


test = AC.Body(1.192, 0.619, 70.8, 282.9, 171.4, 0.0, 'distance')
test.set_epoch(2452700.5)
stream = AC.Stream(test, 10000)

np.savetxt('file.in', stream.get_info(), fmt='%17.9f', delimiter='\t')
