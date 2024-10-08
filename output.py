import numpy as np

def initialize_files( p ):
    p.R_FILE = open("%s/TRAJECTORY.xyz" % (p.DATA_DIR), "w")
    p.V_FILE = open("%s/VELOCITIES.xyz" % (p.DATA_DIR), "w")
    p.E_FILE = open("%s/ENERGIES.dat" % (p.DATA_DIR), "w")
    p.MU_FILE = open("%s/DIPOLE.dat" % (p.DATA_DIR), "w")
    if ( p.do_cavity == True ):
        p.qc_FILE = open("%s/QC.dat" % (p.DATA_DIR), "w")
        p.pc_FILE = open("%s/PC.dat" % (p.DATA_DIR), "w")
    return p

def close_files( p ):
    p.R_FILE.close()
    p.V_FILE.close()
    p.E_FILE.close()
    p.MU_FILE.close()
    if ( p.do_cavity == True ):
        p.qc_FILE.close()
        p.pc_FILE.close()
    return p

def save_data( p, step ):
    if ( step == 0 ):
        p = initialize_files( p )
    
    p.R_FILE.write("%d\n" % p.NATOMS)
    p.R_FILE.write("Step %d\n" % step)
    for at in range( p.NATOMS ):
        p.R_FILE.write("%s %1.5f %1.5f %1.5f\n" % (p.atom_labels[at], p.R_t[step,at,0], p.R_t[step,at,1], p.R_t[step,at,2] ) )
    
    p.V_FILE.write("%d\n" % p.NATOMS)
    p.V_FILE.write("Step %d\n" % step)
    for at in range( p.NATOMS ):
        p.V_FILE.write("%s %1.5f %1.5f %1.5f\n" % (p.atom_labels[at], p.V_t[step,at,0], p.V_t[step,at,1], p.V_t[step,at,2] ) )
    
    p.E_t[step,1] = 0.5 * np.einsum( "a,ad,ad->", p.MASSES, p.V_t[step], p.V_t[step]) # Nuclear Kinetic Energy
    if ( p.do_cavity == False ): 
        p.E_FILE.write("%d %1.5f %1.5f %1.5f\n" % (step, p.E_t[step,0], p.E_t[step,1], np.sum(p.E_t[step,:]) ) )
        
    p.MU_FILE.write("%d %1.5f %1.5f %1.5f\n" % (step, p.MU_t[step,0], p.MU_t[step,1], p.MU_t[step,2] ) )

    if ( p.do_cavity == True ):
        DIP_polarized = np.einsum( "d,d->", p.MU_t[step], p.cavity_polarization)
        p.E_t[step,2]  = 0.5 * p.wc**2 * p.qc_t[step]**2 # Photon Potential Energy
        p.E_t[step,3]  = 0.5 * p.pc_t[step]**2 # Photon Kinetic Energy
        p.E_t[step,4]  = np.sqrt(2 * p.wc**3) * p.A0 * DIP_polarized * p.qc_t[step] # Electron-Photon Interaction Energy        
        p.E_t[step,5] += p.wc * p.A0**2 * DIP_polarized**2 # DSE Energy
        p.E_FILE.write("%d %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f\n" % (step, p.E_t[step,0], p.E_t[step,1], p.E_t[step,2], p.E_t[step,3], p.E_t[step,4], p.E_t[step,5], np.sum(p.E_t[step,:]) ) )
        p.qc_FILE.write("%d %1.5f\n" % (step, p.qc_t[step] ) )
        p.pc_FILE.write("%d %1.5f\n" % (step, p.pc_t[step] ) )

    if ( step == p.NSTEPS - 1 ):
        p = close_files( p )
    
    return p

