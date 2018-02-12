import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
from scipy.constants import e
from scipy.integrate import odeint
from numba import jit
import os
from time import localtime, strftime
import copy

makefigures = True

stime = time.time()

from scipy.constants import e, epsilon_0
def Generation(area, sigma,r):
    #G = a*np.exp(-((x-xstop/2)**2+(y-ystop/2)**2)/(2*c**2))
    normc = 1./(sigma**2*(2.*np.pi)*T)
    G = area*normc*np.exp(-r**2/(2*sigma**2))   # /(cm^3*s)
    return G

import pyqtgraph as pg
import sys
from time import sleep
from PyQt5 import QtCore, QtGui 


showprogress = False

if showprogress:
    app = QtGui.QApplication(sys.argv)
    line_plot = pg.plot()
    line_plot.showGrid(x=True, y=True, alpha=1.)
    
    line_plot2 = pg.plot()
    line_plot2.showGrid(x=True, y=True, alpha=1.)
    
    line_plot3 = pg.plot()
    line_plot3.showGrid(x=True, y=True, alpha=1.)
    
    curv1 = line_plot.plot(pen=None, symbolPen=None, symbolSize=3, symbolBrush=(255, 0, 0))#
    curv2 = line_plot.plot(pen=None, symbolPen=None, symbolSize=3, symbolBrush=(0, 255, 0))#
    curv3 = line_plot2.plot(pen=None, symbolPen=None, symbolSize=3, symbolBrush=(255, 255, 0))#
    curv4 = line_plot3.plot(pen=None, symbolPen=None, symbolSize=3, symbolBrush=(255, 255, 100))#


#from PyQt4 import QtCore, QtGui 

filename = strftime("Normal_%d_%b_%H_%M_%S", localtime())
os.mkdir(filename)
os.mkdir(filename+"/norcharges")
os.mkdir(filename+"/nortotalcharge")
os.mkdir(filename+"/norelectricfield")
os.mkdir(filename+"/nortemperature")
os.mkdir(filename+"/nordt")
os.mkdir(filename+"/norcentertemperature")
os.mkdir(filename+"/norheatsource")
os.mkdir(filename+"/norpotential")


def poissolv(psi, ro, dx2, eps0, M):
    eps = 1+eps0
    ind = 0
    while eps>eps0 and ind<10000:
        newpsi = np.zeros(M)
        
        
        newpsi[1:-1] = (psi[2:]+psi[:-2])/2 + 1./(4*xv[1:-1])*(psi[2:]-psi[:-2])*dx + 0.5*dx2*ro[1:-1]
        
        newpsi[0] = newpsi[1] + 0.5*dx2*ro[0]
        newpsi[-1] = 2*newpsi[-2] - newpsi[-3]  # Ef[-2] = Ef[-1]
        
        eps = max(abs(newpsi - psi))
        #print eps
        psi = newpsi.copy()
        ind+=1
	return psi
    
@jit
def do_timestep_charge(nl, pl, psil,dt):
	n00 = nl[1:-1]
	p00 = pl[1:-1]
	
	ronet = (p-n)*eoveps/T
	eps0 = dx2*max(ronet)/50
	psi = poissolv(psil, ronet, dx2, eps0, xn)
	
	
	
	Ef[1:-1] = -(psil[2:]-psil[:-2])/dx
	Ef[-1] = Ef[-2]
	if max(abs(Ef))!= 0:
		dt =  min(1.*dx2/(8*Dce), 1.*dx/(8*mue*max(abs(Ef))), 1./(8*mue)*max(abs(xv[1:-1]/(Ef[1:-1]*n00))), 1./(8*mue)*max(abs(xv[1:-1]/(Ef[1:-1]*p00))))
	
	Gscld = (gen_values*dt*T)
	D0ecoef = 2*Dce*(dt/dx2)
	D0hcoef = 2*Dce*(dt/dx2)
	Dseclds = Dce*dt/dx2
	Dsecldf = Dce*dt/(2*dx*xv[1:-1])
	Dshclds = Dch*dt/dx2
	Dshcldf = Dch*dt/(2*dx*xv[1:-1])
	Ascld = dt*kt
	Bscld = dt*kr/T
	Cscld = 0.5*dt*ka/T**2
    
    

	GRe = Gscld - Ascld*nl - Bscld*nl*pl - Cscld*(nl**2*pl + nl*pl**2)
	GRh = Gscld - Ascld*pl - Bscld*nl*pl - Cscld*(nl**2*pl + nl*pl**2)
    #print Ef[1:]*n00
    

	
	n[1:-1] = n00 + GRe[1:-1] + Dseclds*( nl[2:] - 2*n00 + nl[:-2] ) + Dsecldf*( nl[2:] - nl[:-2] ) + mue*Ef[1:-1]*(n00/xv[1:-1] +  ( nl[2:] - nl[:-2] ) /2*dx)*dt
	p[1:-1] = p00 + GRh[1:-1] + Dshclds*( pl[2:] - 2*p00 + pl[:-2] ) + Dshcldf*( pl[2:] - pl[:-2] ) - muh*Ef[1:-1]*(p00/xv[1:-1] +  ( pl[2:] - pl[:-2] ) /2*dx)*dt

	n[0] = nl[0] + D0ecoef*(nl[1]-nl[0]) + GRe[0]
	p[0] = pl[0] + D0hcoef*(pl[1]-pl[0]) + GRh[0]
	n[-1] = n[-2]
	p[-1] = p[-2]
	#n[-1] = nl[-1] + GRe[-1] + 2*Dseclds*( nl[-2] - nl[-1]) + mue*Ef[-1]*nl[-1]/xv[-1]*dt
	#p[-1] = pl[-1] + GRh[-1] + 2*Dshclds*( pl[-2] - pl[-2]) - muh*Ef[-1]*pl[-1]/xv[-1]*dt
    
	#ronet = (p-n)*eoveps/T
	#eps0 = dx2*max(ronet)/50
	#psi = poissolv(psil, ronet, dx2, eps0, xn)
	
	last_t = dt
    
	
	return n,p,psi,dt ,Ef
	
def total_recombination(n,p):
	qe = n/T
	qh = p/T

	ple = kr*qe*qh #1/(cm^3*S)
	plh = ple
	nrad_e = kt*qe #1/(cm^3*S)
	nrad_h = kt*qh #1/(cm^3*S)
	augere = ka*qe**2*qh   #1/(cm^3*S)
	augerh = ka*qh**2*qe
	return (sum(ple*xv) + sum(plh*xv) + sum(augere*xv) + sum(nrad_e*xv)+sum(nrad_h*xv)+ sum(augerh*xv))*2*np.pi*dx

def solvediffusion_charge(xn,L, tn, dt, mpnt, tolerance, p0, psi0, n0 = False):
	stime = time.time()
	Ef = np.zeros(xn)
    #n0 = np.zeros(xn, dtype = np.float128 )
	if n0 is None:
		n = np.zeros(xn)
		p = np.zeros(xn)
		psi = np.zeros(xn)
	else:
		n = n0
		p = p0
		psi = psi0	
    
    #n = np.zeros(xn, dtype = np.float128)


    #ntotal = np.zeros(tn/mpnt+1, dtype = np.float128)
	ntotal = np.zeros(tn/mpnt+1)
	tv = np.zeros(tn/mpnt+1)
	tc = 0
	
    #toltotal = np.zeros(tn/mpnt+1)
    

	tol2 = 1
    
	m = 0
	while tol2 > tolerance or m<10**6:
		if m%10 == 0:
			n, p, psi,dt,Ef = do_timestep_charge(n, p, psi,dt)
			tc = tc + dt*10**9
		
		if m>tn:
			print "end by tn, tn is equal to" ,tn, "m is equal to",m
			break
            
		mshort = m/mpnt
		if m%mpnt==0:
			ntotal[mshort] = np.sum(n*xv)
			tv[mshort] = tc 
			if  m>mpnt*2:
				recombination = total_recombination(n,p)*T
				tol2 = abs(1-recombination/(2*curG))

				#tol2 = abs(ntotal[mshort]-ntotal[mshort-2])/(ntotal[mshort]*sum(tv[mshort-2:mshort]*10**3))
				#tol2 = max(abs((nlast-n)/(np.mean(n)))*dt*10**14)
				#print Ef[0],n[0]
		#nlast = n.copy()
		if m%10000==0 and m > 1:
			print 'Current charge tolerance = %1.11f'%tol2
			#print 2*curG,recombination
			if showprogress:
					curv1.setData(n)
					curv2.setData(p)
					curv3.setData(Ef)
					curv4.setData(tv[:mshort-1],ntotal[:mshort-1])
					QtCore.QCoreApplication.processEvents()
			
		#if m> 100000 and tol2 > 0.3:
		#	print "oscillation"
		#	break
		m+=1
	print "real time",m*dt
	print "Charge equilibrium reached ... "  
	print "charge tolerance ", tol2
	print "***************************************\n"
	print "Time taken = %1.5f"%(time.time() - stime), ' s'
	print "Number of iterations = ", m+1, ' out of ', tn, '. Or ', (m+1.)/tn*100.,' %'
    #tn = m #update with a new value
	ntotal = ntotal[:m/mpnt-1]
	tv = tv[:m/mpnt-1]
    #toltotal = toltotal[m-10001:m]
	print "Final time = ", dt*m*10**9, ' ns'

	return n,p,psi, ntotal,Ef,tv

def do_timestep_temperature(ul):

    u[0] = ul[0] + D0coef*(ul[1]-ul[0]) + hbalancedensity[0]*M*dt/(rho*Cp)
    u[1:-1] = ul[1:-1] + Dt1coef*(ul[2:] - 2*ul[1:-1]  + ul[:-2]) +Dt2coef*(ul[2:] - ul[:-2]) + hbalancedensity[1:-1]*M*dt/(rho*Cp)
    u[-1]  =  Tcool
    #ul = u.copy()
    return  u

#def solvediffusion_temperature(xn,L, tn, dt, mpnt, tolerance, u0 = False):
def solvediffusion_temperature(xn,L, tn, dt, mpnt, tolerance):
	stime = time.time()
	'''
	if u0 is None or u0[0] > Tcool:
		
		u = Tcool*np.ones(xn)
	else:
		u = u0
	'''
	u = Tcool*np.ones(xn)
	utotal = np.zeros(tn/mpnt+1)
    
	tol2 = 1
    
	m = 0


	while tol2 > tolerance:
		u = do_timestep_temperature(u)
        
		if m>tn:
			print "end by tn, tn is equal to" ,tn, "m is equal to",m
			break
            
		mshort = m/mpnt
		if m%mpnt==0:
			utotal[mshort] = u[0]
			if  m>mpnt*2:
				tol2 = abs(utotal[mshort] - utotal[mshort-1])/abs(300-utotal[mshort])
		if m%10000==0 and m > 1:
			last_tol = tol2
			print 'Current temperature tolerance = %1.11f'%tol2
			if last_tol < tol2:
				print "oscillation"
				break
		m+=1
	print "real time",m*dt
	print "Temperature equilibrium reached ... "  
	print "temperature tolerance ", tol2
	print "***************************************\n"
	print "Time taken = %1.5f"%(time.time() - stime), ' s'
	print "Number of iterations = ", m+1, ' out of ', tn, '. Or ', (m+1.)/tn*100.,' %'
	#tn = m #update with a new value
	utotal = utotal[:m/mpnt]
    #toltotal = toltotal[m-10001:m]
	print "Final time = ", dt*m*10**9, ' ns'
	return u, utotal

if __name__ == "__main__":
	original_time = time.time()
	dE = 0.1  
	Eg = 1.39

	tn = 5*10**7
	#tolerance_charge = 1*10**(-9)   #10^-8
	tolerance_charge = 0.15
	mpnt = 100 # calculoate tolerance every m points
	Tcool = 300                  #room temperature K
	L = 9*10**(-4)    #Distance between electrodes
	xn = 512              #number of radius steps
	dx = L/xn 
	dx2 = dx**2
	xv = np.linspace(0.,L-dx, xn)        #radius array    
	n = np.zeros(xn)
	p = np.zeros(xn)
	u = Tcool*np.ones(xn)
	psi = np.zeros(xn)
	Ef = np.zeros(xn)
	eps = 13. 						#dielectric constant
	#eps = 10**5						# no field
	mue = 8500.                     #electron mobility cm^2/(V*s)
	muh = 400.						#hole mobility cm^2/(V*s)
	Dce = 200                      #electron diffusivity cm^2/s
	Dch =10							#hole diffusivity cm^2/s
	Dt = 0.31						#heat diffusivity
	
	eoveps = e/(eps*epsilon_0)
	
	M = 144.645     			#g/mol
	rho = 5.32 					#g/cm^3
	Cp = 47.02					#J/(mol*K)
    
	kr = 1.7*10**(-10)             #bimocular recombination rate constant   cm^3/s
	kt = 2.72*10**(5)             # (non-radiative) electron trapping rate constant /s
	ka = 7*10**(-30)                  #Auger recombination rate constant  cm^6/s
    
	#S = 10**(-8)                     #cm^2 
	T = 10**(-5)
	c  = 0.5*10**(-4)                 #standard deviation of laser distribution (twice laser spot radius) cm
	power = 10.**(np.linspace(-1., -4, 16))  #laser power
	gampvalues =  power/e #generation rate range
    
    
    
	
    
    
    
	f = open(filename+'/results.txt','w')
    
	f.write('tolerance = %1.3e\n'%tolerance_charge)
	f.write('L = %1.8f cm - distance between electrodes\n'%L)
	f.write('xn = %1.0f - size of the space grid\n'%xn)
	f.write('epsilon = %1.3f - dielectric permitivity of the material\n'%eps)
    #f.write('mu = %1.3f - cahrge mobilty cm^2/(V*S) \n'%mu)
    #f.write('Dc = %1.3f - cahrge diffusion coefficient cm^2/(S) \n'%Dc)
    
	f.write('kr = %1.4e - bimocular recombination rate constant   cm^3/s \n'%kr)
	f.write('kt = %1.4e - cnon-radiative) electron trapping rate constant 1/s \n'%kt)
	f.write('ka = %1.4e - Auger recombination rate constant  cm^6/s \n'%ka)
    
	f.write('c = %1.7f - standard deviation of laser distribution (twice laser spot radius) cm \n'%c)
    
	f.write('\n')
	f.write('# --------------------------------------------------------\n')
	f.write('\n')
    
    
	f.write('Generation rate ranges from %1.3e '%gampvalues[0] + ' to %1.ef'%gampvalues[-1] + ' 1/(cm^3*s) \n')
	
	f.write('Number of generation rate values =  %1.0f \n'%len(gampvalues))
    
	f.write('\n')
	f.write('# --------------------------------------------------------\n')
	f.write('\n')
    
	plqyvals = np.zeros(len(gampvalues))
	augerqyvals = np.zeros(len(gampvalues))
	nradqyvals = np.zeros(len(gampvalues))
	hbvals = np.zeros(len(gampvalues))
	hbminvals = np.zeros(len(gampvalues))
    
	stime2 = time.time()
	last_n = None
	last_u = None
	last_p = np.zeros(xn)
	last_psi = np.zeros(xn)
	 

	for gind, curG in enumerate(gampvalues):
        #if eind<2:
        #	break
		dt =  1.*dx2/(8*Dce)
		print 'Curent generation rate =  %1.4e W \n'%power[gind]
        
		f.write('Curent generation rate =  %1.4e W\n'%power[gind])
        
        
		gen_values = Generation(curG, c, xv)

        
		Gscld = (gen_values*dt*T)
		D0ecoef = 2*Dce*(dt/dx2)
		D0hcoef = 2*Dce*(dt/dx2)
		Dseclds = Dce*dt/dx2
		Dsecldf = Dce*dt/(2*dx*xv[1:-1])
		Dshclds = Dch*dt/dx2
		Dshcldf = Dch*dt/(2*dx*xv[1:-1])
		Ascld = dt*kt
		Bscld = dt*kr/T
		Cscld = 0.5*dt*ka/T**2
        
		last_n,last_p = np.loadtxt("Normal_12_Feb_01_53_24/norcharges/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt',unpack = True)
		 
		Ef = np.loadtxt("Normal_12_Feb_01_53_24/norelectricfield/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt',unpack = True) 
		#for l in range(xn):
		#	last_psi[l] = (-sum(Ef[:l])+sum(Ef[l:]))*dx
		
		
		n,p,psi, ntotal,Ef, tv = solvediffusion_charge(xn,L, tn, dt, mpnt, tolerance_charge, last_p, last_psi, n0 = last_n)
		#last_n = n       #1/cm^2
		#last_p = p
		#last_psi = psi

		qe = n[:]/T
		qh = p[:]/T
		realtime = len(ntotal)*mpnt*dt*10**9
		f.write('total charge running time = %1.5f ns \n'%realtime)
        
        #PL rate
		pl = kr*qe*qh #1/(cm^3*S)

		nrad_e = kt*qe #1/(cm^3*S)
		nrad_h = kt*qh #1/(cm^3*S)

		auger = 0.5*(ka*qe**2*qh + ka*qh**2*qe) #1/(cm^3*S)
		total_recombine = sum(pl*xv) + sum(auger*xv) + sum(nrad_e*xv)+sum(nrad_h*xv)
		genrate = gen_values[:] # 1/(cm^3*s)
		plqyvals[gind] = np.sum(pl*xv)/total_recombine
		augerqyvals[gind]  = np.sum(auger*xv)/total_recombine
		nradqyvals[gind]  = (np.sum(nrad_e*xv)+np.sum(nrad_h*xv))/total_recombine

		Auheat = auger*Eg*e                # W/cm^3
		nradheat = (nrad_e+nrad_h)*Eg*e    # W/cm^3

		Heating = Auheat + nradheat # W/cm^3
		Cooling = genrate*dE*e              # W/cm^3

		hbvals[gind]  = np.sum(Cooling)/np.sum(Heating)	
        
		hbalancedensity = Heating - Cooling
		hbminvals[gind] = hbalancedensity[0]
        
		dt =  1*dx2/(4*Dt)
		Dt1coef = Dt*dt/dx2   #second order derivative term
		Dt2coef = Dt*dt/(2*dx*xv[1:-1])            #first order derivative term
		D0coef = 2*Dt*(dt/dx2) 
		
		tolerance_temperature = 10**(-7)
		
		#mpnt = 500
		#u, utotal = solvediffusion_temperature(xn,L, tn, dt, mpnt, tolerance_temperature, u0 = last_u)
		#u, utotal = solvediffusion_temperature(xn,L, tn, dt, mpnt, tolerance_temperature)
		#last_u = u
        
        
		charges = zip(n,p)
		heating_cooling = zip(Heating,Cooling)
		#realtime = len(utotal)*mpnt*dt*10**9
		f.write('total temperature real time = %1.5f ns \n'%realtime)
		
		if showprogress:
			app.exec_()
		if makefigures:
			plt.plot(xv,qe,'b', linewidth=2)
			plt.plot(xv,qh,'r', linewidth=2)
			plt.savefig(filename+"/norcharges/"+"_%03d_"%gind +"charge_distribution"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
			
			
			plt.plot(xv,Heating,'r', linewidth=2)
			plt.plot(xv,Cooling,'b', linewidth=2)
			plt.plot(xv,hbalancedensity,'g', linewidth=2)
			plt.savefig(filename+"/norheatsource/"+"_%03d_"%gind +"charge_distribution"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
            
            
			plt.plot(tv, ntotal, linewidth=2)
			plt.savefig(filename+"/nortotalcharge/"+"_%03d_"%gind + "total_charge"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
            
            
			plt.plot(xv, Ef, linewidth=2)
			plt.savefig(filename+"/norelectricfield/"+"_%03d_"%gind + "total_charge"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
			'''
			plt.plot(xv, u, linewidth=2)
			plt.savefig(filename+"/nortemperature/"+"_%03d_"%gind + "total_charge"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
            
            
			plt.plot(utotal, linewidth=2)
			plt.savefig(filename+"/norcentertemperature/"+"_%03d_"%gind + "total_charge"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
			'''
			plt.plot(tv, linewidth=2)
			plt.savefig(filename+"/nordt/"+"_%03d_"%gind + "total_charge"+'_grate=%1.2e.png'%power[gind], dpi=400)
			plt.close()
		np.savetxt(filename+"/nortotalcharge/"+"_%03d_"%gind + '_grate=%1.2e.png'%power[gind]+'.txt',ntotal)
		np.savetxt(filename+"/norcharges/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', charges)
		np.savetxt(filename+"/norelectricfield/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', Ef)
		np.savetxt(filename+"/norpotential/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', psi)
		#np.savetxt(filename+"/nortemperature/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', u)
		np.savetxt(filename+"/nordt/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', tv)
		np.savetxt(filename+"/norheatsource/"+"_%03d_"%gind + '_grate=%1.2e'%power[gind]+'.txt', heating_cooling)
		f.write("total_recombination=%1.3f"%total_recombine)
		f.write("total_generation=%1.3f"%curG)
       
	print " "
	print "Time taken for the whole thing = ", time.time() - stime2
    
	np.savetxt(filename+'/plqyvals.txt', plqyvals)
	np.savetxt(filename+'/augerqyvals.txt', augerqyvals)
	np.savetxt(filename+'/nradqyvals.txt', nradqyvals)
	np.savetxt(filename+'/hbvals.txt', hbvals)
	np.savetxt(filename+'/hbminvals.txt', hbminvals)
    
	np.savetxt(filename+'/gampvalues.txt', gampvalues)
    
	time_spend = time.time() - original_time
    
	print "runing time is ", time_spend
	f.write("runing_time=%1.3f"%time_spend)
	f.close()



