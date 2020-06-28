"""
Basic atmosphere functions
This dark flight model predicts the landing sight of a meteoroid by
propagating the position and velocity through the atmosphere using a
5th-order adaptive step size integrator (ODE45).

Created on Mon Oct 17 10:59:00 2016
@author: Trent Jansen-Sturgeon
"""
import numpy as np
from scipy.interpolate import interp1d

def reynolds(rho, vel, dvisc, length=0.05):
    """returns reynolds no as fn of fluid density, 
    velocity, dynamic viscocity and diam of body"""
    #put len as 0.05, for all typical meteorites
    return rho * vel * length / dvisc 

def knudsen(mach, re):
    """returns knudsen number, fn of mach and reynolds, for dry air
    (take ratio specific heats as 1.40)"""

    # Note: 1.4 is only specifically for dry air at atmospheric temperatures. 
    # We might need to change this in darkflight...

    return mach / re * np.sqrt(np.pi * 1.40 / 2.0) 

def SoS(T):
    """function input atmospheric temp (T). returns speed of sound in m/s
    as fn of temperature (in K).
    see:
    http://en.wikipedia.org/wiki/Earth's_atmosphere and
    http://en.wikipedia.org/wiki/Density_of_air
    """
    return 331.3 * np.sqrt( T / 273.15)

def viscosity(T):
    """function input atmospheric T in Kelvin. Dynamic Viscosity of atm in Pa.s
       is returned.
       Function uses Sutherland's formula.
    """
    ###viscosty of air as function of height, using Sutherland's formula:
    #http://en.wikipedia.org/wiki/Viscosity#Gases
    #http://wiki.xtronics.com/index.php/Viscosity#The_dependence_on_pressucd_of_Viscosity
    #('With Gases until the pressure is less than 3% of normal air pressure
    #the change is negligible on falling bodies.')

    C = 120  #Sutherland's constant for air - Kelvin

    #see wikipedia http://en.wikipedia.org/wiki/Viscosity
    # Gas   C[K]      T_ref[K]   mu_ref[mu_Pa s]
    # air   120       291.15    18.27

    reference_visc = 18.27e-6
    reference_T = 291.15

    return reference_visc * (reference_T + C)/(T + C) * (T / reference_T)**1.5

def dragcoef(re, mach, kn=0.0001, A=1.21):
    """returns drag coefficient as fn of reynolds number, mach and knudsen no;
    for a spherical particle in fluid

    """
    ##Kn >=10 : ReVelle 1976; Masson et al. 1960
    ## Transition Flow Regime : 0.1<Kn<10 : Khanukaeva 2005
    ## Continuum Flow Regime : Kn >= 0.1
    ##      -Hypersonic: for spheres  (Bronshten 1983, 1983; ReVelle 1976; Masson et al. 1960)
    ##                   for circular cylinders (Truitt 1959)
    ##                   for tiles and bricks (Zhdan et al. 2007)
    ##      -Subsonic (see Haider and Levenspiel (1989, 1989))


    cd_fm = 2.00
    cd_cont = 0.98182 * A**2 - 1.78457 * A + 1.641837
    cd_mid = 1.0
    cd_low = 1.0
    cd_rescale = 1.0 #rescale low speed drag as ratio compared to sphere
    kn_max = 10.0 # free molecular flow
    kn_min = 0.01 #continuum regime

    if kn > kn_max: 
        cd = cd_fm

    elif kn > kn_min: # bridging function for transition to continuum
        # top = np.log( 1.0/kn) - np.log( 1.0/kn_max)
        # bot = np.log( 1.0/kn_min) - np.log( 1.0/kn_max)
        # avg = (cd_fm + cd_cont) / 2.0
        # diff = (cd_fm - cd_cont) / 2.0
        # cd = avg + diff * np.exp( np.pi*top/bot )
        cd = cd_cont + (cd_fm - cd_cont) * np.exp(-0.001 * re**2)
        #return 1.55 + 0.65 * np.cos( np.pi*top/bot )
    else:   #now in pure contimuum flow, **the default for most darkflight**
        if mach > 8.0:
            cd = cd_cont
##            if A<0.75:
##                cd=2.0;
##            elif A<1.21:
##                A_min = 0.75;
##                cd=0.92 + 1.08 * (1.21-A) / (1.21-A_min);
##            elif A<1.75:
##                A_max = 1.75;
##                cd=0.92 + 1.08 * (A - 1.21) / (A_max- 1.21);
##            else:
##                cd=2.0;

        elif mach > 2.0: # 2 to 8
            #arbitrary bridging function
            cd = cd_mid + (cd_cont-cd_mid) * (mach-2.0) / (8.0-2.0)   
        elif mach > 1.1: # 1.1 to 2
            cd = cd_mid
        elif mach > 0.6: #0.6 to 1.1
            cd = cd_low * (0.725 + 0.275*np.sin( np.pi*(mach-0.85)/0.5))
        else:    # mach < 0.6:
            # a = 0.1806; b = 0.6459
            # c = 0.4251; d = 6880.95
            # cd = 24 / re * (1 + a * re**b) + c / (1 + d / re) 

            # # re cutt-ofs from boundary layer theory
            if re< 2e5:
                # from Brown&lawler
                cd=((24/re)*(1.0+0.15*re**0.681))+(0.407/(1+8710/re));

            elif re<3.2e5:
                cd =- 2.24229520715170e-17*re**3 \
                    + 2.28663611400439e-11*re**2 \
                    - 7.46988882625855e-06*re \
                    + 0.986662115581471

            elif re<3.5e6:
                cd =0.4/3180000*re+0.15974842767295

            else:
                cd = 0.6

    return cd

######################################

def interp_shape(A, vals):
    # Shape (A) corresponding to [sphere, cylinder, brick]
    A_vals = np.array([1.21, 1.6, 2.7])
    val = interp1d(A_vals, vals, kind='linear', fill_value='extrapolate')(A)
    return val

def cd_hypersonic(A):

    # Hypersonic drag coefficient for a variety of shapes
    cd_hyp_vals = np.array([0.92, 1.3, 2.0])
    cd_hyp = interp_shape(A, cd_hyp_vals)
    return cd_hyp

def cd_subsonic(re, A):

    ''' Sub-critical drag coefficient '''
    # Estimate thi assuming an ellipsoid
    V = 1; sa_eq = (36 * np.pi * V**2)**(1./3)
    a_ax = np.sqrt(A * V**(2./3) / np.pi); c_ax = (3 * V**(1./3)) / (4 * A)
    thi = sa_eq / (4 * np.pi * ((a_ax**3.2 + 2 * (a_ax * c_ax)**1.6) / 3)**(1./1.6))
    thi_perp = (sa_eq / 4) / (np.pi * a_ax**2)

    # # Equation 10 of Holzer and Sommerfeld (2008)
    # cd_sub = lambda re: 8./(re * thi_perp**0.5) + 16./(re * thi**0.5) +\
    #      + 3./(re**0.5 * thi**0.75) + 0.4210**(0.4 * (-np.log(thi))**0.2) / thi_perp

    # Sub-critical regime - Haider and Levenspiel (1989)
    a = np.exp(2.3288 - 6.4581 * thi + 2.4486 * thi**2)
    b = 0.0964 + 0.5565 * thi
    c = np.exp(4.905 - 13.8944 * thi + 18.4222 * thi**2 - 10.2599 * thi**3)
    d = np.exp(1.4681 + 12.2584 * thi - 20.7322 * thi**2 + 15.8855 * thi**3)
    cd_subcrit = lambda re: 24./re * (1 + a * re**b) + c/(1 + d/re)

    # Forget about the critical / supercritical regions for now...
    cd_sub = cd_subcrit(re)

    # ''' Super-critical drag coefficient '''
    # cd_super_vals = np.array([0.33, 0.6, 2.0]) #<--- need to change this to cd_supercrit
    # cd_supercrit = interp_shape(A, cd_super_vals)

    # ''' Linking drag coefficient through the critical region '''
    # # Magic correction function 1: Logistic fn (in logspace)
    # logistic_fn = lambda re: cd_subcrit(re) + (cd_supercrit - cd_subcrit(re)) \
    #         / (1 + (los*re_c / re)**(1/log_hw))
    
    # # # Smooth bodies - Boundary Layer Theory by Schlichting 
    # # log_hw = 0.6 # log-half-width of the critical region
    # # re_c = 4.5e5 # re at cd_critical
    # # los = 2. # logistic function offset
    # # cd_dip = [0.1, 0.3, logistic_fn(re_c)] # lowest point within cd_critical for sphere/cylinder
    
    # # Rough bodies - 
    # log_hw = 0.4 # log-half-width of the critical region
    # re_c = 1e5 # re at cd_critical
    # los = 2. # logistic function offset
    # cd_dip = [0.2, 0.5, logistic_fn(re_c)] # lowest point within cd_critical for sphere/cylinder

    # # Magic correction function 2: Gumbel Distribution (in logspace)
    # cd_c = interp_shape(A, np.array(cd_dip)) # cd_critical value
    # gumbel_dist = lambda re: (cd_c - logistic_fn(re_c)) * \
    #         np.exp(1 - 1/log_hw * np.log(re/re_c) - (re_c/re)**(1/log_hw))
    
    # # Overall corrected drag equation
    # cd_sub = logistic_fn(re) + gumbel_dist(re)

    return cd_sub

def cd_fm(vel):
    
    # Calculation of Coefficients in Meteoric Physics Equations - Khanukaeva (2005)
    cd_fm = 2. + np.sqrt(1.2) / (2. * vel/1000.) * (1. + (vel/1000.)**2 / 16. + 30.)

    return cd_fm

def dragcoeff(vel, temp, rho_a, A): # by Trent
    '''
    The drag coefficient is bloody annoying!
    '''

    # Determine some needed variables
    mu_a = viscosity(temp) # Air Viscosity (Pa.s)
    mach = vel / SoS(temp) # Mach Number
    re = reynolds(rho_a, vel, mu_a, 0.1) # Reynolds Number
    kn = knudsen(mach, re) # Knudsen Number

    ''' Determine cd '''
    if kn > 10.0: # Free molecular flow 
        cd = cd_fm(vel)
        # ^---- still not fully convinced by this equation - never gets near the traditional value of 2?
    
    elif kn > 0.01: # Bridging function for transition to continuum
        cd = cd_subsonic(re, A) + (cd_fm(vel) - cd_subsonic(re, A)) * np.exp(-0.001 * re**2)
        # ^---- don't know how to verify yet
        # print('Transition region: kn={0:.2e}, cd={1:.2f}'.format(kn, cd))
    
    else:   # Pure continuum flow [the default for most darkflight]
        # See Miller and Bailey (1979) for details
        cd_sub = cd_subsonic(re, A)
        cd_hyp = cd_hypersonic(A)

        logistic_fn = lambda M: cd_sub + (cd_hyp - cd_sub) \
            / (1 + np.exp(-(M - M_c) / hw))

        # Can tweek these perameters
        hw_vals = np.array([0.5, 0.3, 0.1]) # logistic and gumbel half-widths
        hw = interp_shape(A, hw_vals)
        M_c_vals = np.array([1.5, 1.2, 1.1])
        M_c = interp_shape(A, M_c_vals)
        cd_crit_vals = np.array([1, logistic_fn(mach)/0.92, logistic_fn(mach)/0.92])# Critical values
        cd_c = interp_shape(A, cd_crit_vals)

        gumbel_dist = lambda M: (cd_c - logistic_fn(M))* np.exp(-(M - M_c) / hw \
            - np.exp(-(M - M_c) / hw)) / np.exp(-1)

        cd = logistic_fn(mach) + gumbel_dist(mach)

        # # Miller and Bailey (1979) - page12
        # mmach = np.array([0.3, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2, 3, 4])
        # cd_mm = np.array([-1, 0.55, 0.6, 0.65, 0.68, 0.75, 0.85, 0.98, 1.0, 0.97, 0.95, -1]) 
        # print(logistic_fn(mm) + gumbel_dist(mm))

    return cd, re, kn, mach

def dragcoefff(m, A):
    # Drag coefficient according to Carter, et.al. (2011)
    # More to be used as upper and lower limits for cd 
    m = float(m)

    # Cube (A = 2.18)
    if m >= 1.150:
        cd_cube = 2.1 * np.exp(-1.16 * (m+0.35)) - 6.5 * np.exp(-2.23 * (m+0.35)) + 1.67
    elif m >= 0:
        cd_cube = 0.60 * m**2 + 1.04

    # Sphere (A = 1.21)
    if m >= 0.722:
        cd_sphere = 2.1 * np.exp(-1.2 * (m+0.35)) - 8.9 * np.exp(-2.2 * (m+0.35)) + 0.92
    elif m >= 0:
        cd_sphere = 0.45 * m**2 + 0.424

    shapes = np.array([1.21, 2.18]); cd_shape = np.array([cd_sphere, cd_cube])
    cd = interp1d(shapes, cd_shape, kind='linear', fill_value='extrapolate')(A)

    return cd


# Test the drag coefficient:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    temp = 300.; rho_a = 1.225 # At sea-level 
    vel = np.logspace(1, 4, 1000)
    A = np.linspace(1.21,2.7,5)
    fig, axs = plt.subplots(2,1,figsize=(16,9))


    # # Plot the original equations ==========================================
    # mu_a = viscosity(temp) # Air Viscosity (Pa.s)
    # mach = vel / SoS(temp) # Mach Number
    # for a in A:
    #     re = reynolds(rho_a, vel, mu_a, 0.1) # Reynolds Number
    #     kn = knudsen(mach, re) # Knudsen Number
    #     cd = np.zeros(len(vel))
    #     for i in range(len(vel)):
    #         cd[i] = dragcoef(re[i], mach[i], kn[i], a)

    #     axs[0].plot(re, cd, '-', label=str(a))
    #     axs[1].plot(mach, cd, '-', label=str(a))


    # # Plot the new equations =============================================
    for a in A:
        cd = np.zeros(len(vel))
        re = np.zeros(len(vel))
        kn = np.zeros(len(vel))
        mach = np.zeros(len(vel))
        for i, v in enumerate(vel):
            [cd[i], re[i], kn[i], mach[i]] = \
            dragcoeff(v, temp, rho_a, a)

        axs[0].plot(re, cd, '-', label=str(a))
        axs[1].plot(mach, cd, '-', label=str(a))


    axs[0].set_xscale('log')#; axs[0].set_yscale('log')
    axs[0].set_xlim([1e4,1e8]); axs[0].set_ylim([0,4])
    axs[1].set_xlim([0,5]); axs[1].set_ylim([0,4])
    axs[0].set_xlabel('Reynolds'); axs[0].set_ylabel('cd')
    axs[1].set_xlabel('Mach No.'); axs[1].set_ylabel('cd')
    axs[0].legend(); axs[1].legend(); axs[0].grid(); axs[1].grid()
    fig2, axs = plt.subplots(2,1,figsize=(16,9))


    mach = np.hstack((np.linspace(0.2,1.2,6), np.linspace(1.5,4,6)))
    for m in mach:
        v = m * SoS(temp)
        RE = np.logspace(4,7,1000)
        Rho_a = RE * viscosity(temp) / (v * 0.1)
        cd = np.zeros(len(vel))
        re = np.zeros(len(vel))
        kn = np.zeros(len(vel))
        mach = np.zeros(len(vel))
        for i, rho_a in enumerate(Rho_a):
            [cd[i], re[i], kn[i], mach[i]] = \
                dragcoeff(v, temp, rho_a, 1.21)

        axs[0].plot(re, cd, '-', label=str(m))
        axs[1].plot(mach, cd, '.', label=str(m))


    # Plot the new new equation ===========================================
    # T = 300; dvisc = viscosity(T)
    # mach = vel / SoS(T)
    # for a in A:
    #     cd = np.zeros(len(mach))
    #     for i, m in enumerate(mach):
    #         cd[i] = dragcoefff(m, a)

    #     axs[1].plot(mach, cd, '-', label=str(a))

    #     re = reynolds(rho_a, vel, dvisc, 0.1)
    #     axs[0].plot(re, cd, '-', label=str(a))



    axs[0].set_xscale('log')#; axs[0].set_yscale('log')
    axs[0].set_xlim([1e4,1e8]); axs[0].set_ylim([0,4])
    axs[1].set_xlim([0,5]); axs[1].set_ylim([0,4])
    axs[0].set_xlabel('Reynolds'); axs[0].set_ylabel('cd')
    axs[1].set_xlabel('Mach No.'); axs[1].set_ylabel('cd')
    axs[0].legend(); axs[1].legend(); axs[0].grid(); axs[1].grid()
    plt.show()




    # # Finding the relation between thi and shape [thi(A)]:
    # plt.figure()
    # M = 10.; rho = 3500.; V = M / rho
    # sa_eq = (36 * np.pi * V**2)**(1./3)
    # A = np.linspace(1,3,100)

    # a = np.sqrt(A * V**(2./3) / np.pi)
    # c = (3 * V**(1./3)) / (4 * A)
    # thi_elip = sa_eq / (4 * np.pi * ((a**3.2 + 2 * (a * c)**1.6) / 3)**(1./1.6))
    # plt.plot(A, thi_elip, label='ellipsoid')

    # a = np.sqrt(A * V**(2./3))
    # c = V**(1./3) / A
    # thi_cube = sa_eq / (2 * (a**2 + 2 * a * c))
    # plt.plot(A, thi_cube, label='cube')

    # r = (2 * V**(1./3)) / (np.pi * A)
    # h = V / (np.pi * r**2)
    # thi_cyli = sa_eq / (2 * np.pi * r * h + 2 * np.pi * r**2)
    # plt.plot(A, thi_cyli, label='cylinder')

    # plt.axvline(x=1.21)
    # plt.xlabel('Shape'); plt.ylabel('thi')
    # plt.legend(); plt.show()

