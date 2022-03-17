from pymatbridge import Matlab
matlabpath = '/usr/local/bin/matlab'
mlab = Matlab(executable= matlabpath)
mlab.start()
results = mlab.run_code('a=1;')
print(results)
mlab.stop()