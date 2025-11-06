#This module contains user defined functions (casadi) that which relate volumetric moisture, capillary capacity, and the hydraulic conductivity with pressure head (h)
from casadi import *
def volumetricMoisture(head, soilPars):
	#Dependence of volumetric moisture content on the pressure head
	Se = if_else(head >= 0.,1.,(1 + fabs(head*soilPars['alpha'])**soilPars['n'])**(-soilPars['m']))
	volMoisture = soilPars['thetaR'] + (soilPars['thetaS'] - soilPars['thetaR'])*Se
	return volMoisture

def capillaryCapacity(head, soilPars):
	#Dependence of the capillary capacity on the pressure head
	Se = if_else(head >= 0.,1.,(1+fabs(head*soilPars['alpha'])**soilPars['n'])**(-soilPars['m']))
	dSedh = soilPars['alpha']*soilPars['m']/(1 - soilPars['m'])*Se**(1/soilPars['m'])*(1-Se**(1/soilPars['m']))**soilPars['m']
	capCapacity = Se*soilPars['Ss']+(soilPars['thetaS']-soilPars['thetaR'])*dSedh
	return capCapacity

def hydraulicConductivity(head, soilPars):
	#Dependence of unsaturated hydraulic conductivity on the pressure head
	Se=if_else(head>=0.,1.,(1+fabs(head*soilPars['alpha'])**soilPars['n'])**(-soilPars['m']))
	hyrdConductivity = soilPars['Ks']*Se**soilPars['neta']*(1-(1-Se**(1/soilPars['m']))**soilPars['m'])**2
	return hyrdConductivity

def pressureHead(volMoisture,soilPars):
	#Dependence of pressure head on the volumetric moisture content
	presHead = (((((volMoisture - soilPars['thetaR']) / (soilPars['thetaS'] - soilPars['thetaR'] + 1.e-20) + 1.e-20) ** (1. / (-(1-1/(soilPars['n']+1.e-20)) + 1.e-20))
              - 1) + 1.e-20) ** (1. / (soilPars['n'] + 1.e-20))) / (-soilPars['alpha'] + 1.e-20)
	return presHead

def hydraulicConductivityApprox(head_1, head_2, soilPars):
	#The approximation of the hydraulic conductivity at the center of the compartment
	hydCondApprox = 0.5*(hydraulicConductivity(head_1, soilPars) + hydraulicConductivity(head_2, soilPars))
	return hydCondApprox