import numpy as np
import matplotlib.pyplot as plt

def borehole_ac2d(vpmodel, nx, nz, nt, dx, dt, isx, isz, irx, irz, ist, f0, nop=5):
  """
  Borehole 2D Acoustic Finite-Difference Simulation

  INPUT:

  vpmodel: Velocity model
  nx, nz: Grid points in x and z
  nt: Number of time steps
  dx: Grid increment
  dt: Time step
  isx, isz: Source index in x and z
  ist: Shifting of source time function
  irx, irz: Array of receiver indexes in x and z
  f0: Dominant frequency of source (Hz)
  nop: length of operator (3 for 3-point and 5 for 5-point operator). Default is 5.

  OUTPUT:

  pnew: Updated pressure wavefield. Has shape of (nt, dx, dz)
  seis: Seismogram array. Has shape of (m, nt), where n is number of geophones
  """
  
  # Initial pressure field and its 2nd derivative for each direction
  p = np.zeros((nz, nx))
  pold = np.zeros((nz, nx))
  pnew = np.zeros((nz, nx))
  pxx = np.zeros((nz, nx))
  pzz = np.zeros((nz, nx))  

  # Source time function Gaussian
  src = np.empty(nt + 1)
  T = 1 / f0 # Period
  for it in range(nt):
      src[it] = np.exp(-1.0 / T ** 2 * ((it - ist) * dt) ** 2)

  src = np.diff(src) / dt
  src[nt - 1] = 0

  # Simulation
  pnew = []
  for it in range(nt):
      if nop==3:
          # calculate partial derivatives, be careful around the boundaries
          for i in range(1, nx - 1):
              pzz[:, i] = p[:, i + 1] - 2 * p[:, i] + p[:, i - 1]
          for j in range(1, nz - 1):
              pxx[j, :] = p[j - 1, :] - 2 * p[j, :] + p[j + 1, :]

      if nop==5:
          # calculate partial derivatives, be careful around the boundaries
          for i in range(2, nx - 2):
              pzz[:, i] = -1./12*p[:,i+2]+4./3*p[:,i+1]-5./2*p[:,i]+4./3*p[:,i-1]-1./12*p[:,i-2]
          for j in range(2, nz - 2):
              pxx[j, :] = -1./12*p[j+2,:]+4./3*p[j+1,:]-5./2*p[j,:]+4./3*p[j-1,:]-1./12*p[j-2,:]        
              
      pxx /= dx ** 2
      pzz /= dx ** 2

      # Time extrapolation
      pnew_ = 2 * p - pold + dt ** 2 * vpmodel ** 2 * (pxx + pzz)
      # Add source term at isx, isz
      pnew_[isz, isx] = pnew_[isz, isx] + src[it]

      pold, p = p, pnew_
      pnew.append(pnew_)

      # Save seismograms
      ir = np.arange(len(irx))
      seis = np.zeros((len(irx), nt)) # Initial seismogram as zeros
      seis[ir, it] = p[irz[ir], irx[ir]]

  return pnew, seis

class velmodel:
  """
  Create velocity model of formation and wellbore
  
  INPUT:
  nx, nz: Grid points in x and z  

  This class has 10 functions:
  * `homogeneous`: Homogeneous model (no wellbore). c0 is the velocity.
  * `invadedzone`: Invaded zone model. c_invade is the velocity of invaded zone.
  * `mud`: Mud filtrate in cased-hole model. c_mud is velocity of mud filtrate.
  * `casing`: Casing in cased-hole model. c_casing is velocity of steel casing.
  * `cement`: Cement in cased-hole model. c_cement is velocity of cement.
  * `openhole`: Open-hole model. c_drfl is velocity of drilling fluid. 
                This model does not have `casing` and `cement` elements
  * `owc`: Oil-water contact model. Inputs;
    - c_oilsat is Velocity of oil-saturated zone
    - location is Location of OWC at grid point nz
  * `washout`: Washout model. c_cement is velocity of cement.
  * `fracture`: Horizontal fracture model. Inputs;
    - c_lowvel is Velocity of fractured zone
    - location is Location of fracture at grid point nz
  * `laminae`: Shale lamination model. Inputs;
    - c_laminae is Velocity of laminating layer
    - location is Location of laminae at grid point nz
    - thickness is Thickness of laminae in grid point nz unit
  """
  def __init__(self, nx, nz):
    self.nx = nx
    self.nz = nz  
  
  def homogeneous(self, c0):
    vel = np.zeros((self.nz, self.nx))
    vel += c0
    self.vel = vel
    return self.vel
  
  def invadedzone(self, c_invade):
    self.vel[:, self.nx // 2 - 15: self.nx // 2 + 15] = c_invade    
    return self.vel

  def mud(self, c_mud):
    self.vel[:, self.nx // 2 - 9: self.nx // 2 + 9] = c_mud
    return self.vel

  def casing(self, c_casing):
    self.vel[:, self.nx // 2 - 7: self.nx // 2 + 7] = c_casing
    return self.vel

  def cement(self, c_cement):
    self.vel[:, self.nx // 2 - 5: self.nx // 2 + 5] = c_cement
    return self.vel

  def openhole(self, c_drfl):
    self.vel[:, self.nx // 2 - 7: self.nx // 2 + 7] = c_drfl
    return self.vel

  def owc(self, c_oilsat, location):
    self.vel[:location, :] = c_oilsat 
    return self.vel   
  
  def washout(self, c_cement):
    self.vel[60:80, self.nx // 2 - 20: self.nx // 2 - 9] = c_cement
    self.vel[60:80, self.nx // 2 + 9: self.nx // 2 + 20] = c_cement
    return self.vel
  
  def fracture(self, c_lowvel, location):
    self.vel[location:location+1,:] = c_lowvel
    return self.vel
  
  def laminae(self, c_laminae, location, thickness):
    self.vel[location:location+thickness,:] = c_laminae
