import h5py
import numpy as np

FILE = "eminus_Ele-Eta0-Phi0-Energy50.h5"

def run():

   # Initialize the data.
   DIM0 = 3000
   DIM1 = 1
   energydata = np.zeros((DIM0, DIM1), dtype=np.float64)
   for i in range(DIM0):
      for j in range(DIM1):
         energydata[i][j] = 50.0

   DIM2 = 6
   DIM3 = 360
   cellsdata = np.zeros((3000,DIM2*DIM3), dtype=np.float64)
   cellsdata = np.loadtxt("../../data/GEN-SIM_Ele-Eta0-Phi0-Energy50.txt",dtype=np.float64)
   print(cellsdata.shape)
   caloCells = cellsdata.reshape(DIM0,DIM2,DIM3)
   print(caloCells.shape)
   for i in range(0,6):
      print(caloCells[0][i][0:12])
   
   with h5py.File(FILE, 'w') as f:
      energyDset = f.create_dataset("energy", (DIM0, DIM1), h5py.h5t.IEEE_F64LE)
      energyDset[...] = energydata
      caloCellsDset = f.create_dataset("layer_0", (DIM0, DIM2, DIM3), h5py.h5t.IEEE_F64LE)
      caloCellsDset[...] = caloCells
      
if __name__ == "__main__":
    run()
