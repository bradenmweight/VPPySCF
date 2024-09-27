import numpy as np
from matplotlib import pyplot as plt

from pyscf_tools import do_el_structure, do_dipole_gradient, do_dipole_gradient_diatomic_z
from linear_spectroscopy import get_IR, get_Transmission_Sectrum
from parameters import Parameters
from output import save_data


def run_nuclear( p ):

    def get_energies_gradients( p, step, R ):
        E, DIP, E_GRAD = do_el_structure( p.atom_labels, R, p.el_structure_method )
        p.MU_t[step] = DIP

        p.E_t[step,0] = E # Electronic Energy

        return p, -E_GRAD

    # Do first electronic structure calculation
    p, F1 = get_energies_gradients( p, 0, p.R_t[0] )
    p = save_data( p, 0 )
    for i in range( 1, p.NSTEPS ):
        print( "Step %d of %d" % (i, p.NSTEPS) )
        p.R_t[i]                   = p.R_t[i-1] + p.dtI*p.V_t[i-1] + 0.500 * p.dtI**2 * F1 / p.MASSES[:,None]
        p, F2                      = get_energies_gradients( p, i, p.R_t[i] )
        p.V_t[i]                   = p.V_t[i-1] + 0.500 * p.dtI * (F1 + F2) / p.MASSES[:,None]
        F1                         = F2
        p = save_data( p, i )
        
        print( "dE (Error) = %1.10f" % (np.sum( p.E_t[i,:] ) - np.sum( p.E_t[i-1,:] )) )

    return p

def run_cavity( p ):

    def get_energies_gradients( p, step, R, qc ):
        E, DIP, E_GRAD = do_el_structure( p.atom_labels, R, p.el_structure_method )
        p.MU_t[step,:] = DIP
        DIP_polarized  = np.einsum( "...d,d->...", DIP, p.cavity_polarization)
        p.E_t[step,0]  = E # Electronic Potential Energy

        F_R           = -E_GRAD #-1 * get_nuclear_cavity_gradient( p, step, R, qc, E_GRAD, DIP_polarized )
        F_qc          = 0 * -1 * get_photon_cavity_gradient( p, step, qc, DIP_polarized )
        return p, F_R, F_qc

    def get_nuclear_cavity_gradient( p, step, R, qc, E_GRAD, DIP_polarized ):
        if ( p.A0 == 0.0 ):
            return E_GRAD
        DIP_GRAD       = do_dipole_gradient( p.atom_labels, R, p.el_structure_method, is_diatomic=p.is_diatomic )
        DIP_GRAD       = np.einsum( "...d,d->...", DIP_GRAD, p.cavity_polarization)
        GRAD_elph      = np.sqrt(2 * p.wc**3) * p.A0 * qc * DIP_GRAD
        GRAD_dse       = p.wc * p.A0**2 * 2 * DIP_polarized * DIP_GRAD # Factor of 2 from chain rule
        return E_GRAD + GRAD_elph + GRAD_dse

    def get_photon_cavity_gradient( p, step, qc, DIP_polarized ):
        return p.wc**2 * qc + np.sqrt(2 * p.wc**3) * p.A0 * DIP_polarized

    # Do first electronic structure calculation
    p, F1_R, F1_qc = get_energies_gradients( p, 0, p.R_t[0], p.qc_t[0] )
    p = save_data( p, 0 )

    # Dynamics Loop
    for i in range( 1, p.NSTEPS ):
        print( "Step %d of %d" % (i, p.NSTEPS) )
        # UPDATE POSITIONS
        p.R_t[i]  = p.R_t [i-1] + p.dtI*p.V_t [i-1] + 0.500 * p.dtI**2 * F1_R / p.MASSES[:,None]
        p.qc_t[i] = p.qc_t[i-1] + p.dtI*p.pc_t[i-1] + 0.500 * p.dtI**2 * F1_qc

        # GET NEW FORCES
        p, F2_R, F2_qc = get_energies_gradients( p, i, p.R_t[i], p.qc_t[i] )

        # UPDATE VELOCITIES
        p.V_t[i]  = p.V_t [i-1] + 0.500 * p.dtI * (F1_R  + F2_R)  / p.MASSES[:,None]
        p.pc_t[i] = p.pc_t[i-1] + 0.500 * p.dtI * (F1_qc + F2_qc)

        # PREPARE FOR NEXT ITERATION
        F1_R               = F2_R
        F1_qc              = F2_qc
        p = save_data( p, i )

        print( "dE (Error) = %1.10f" % (np.sum( p.E_t[i,:] ) - np.sum( p.E_t[i-1,:] )) )


    return p



if ( __name__ == "__main__" ):
    params = Parameters()
    
    if ( params.do_cavity == False ):
        # BARE ELECTRONIC DYNAMICS
        params = run_nuclear( params )
    else:
        # CAVITY DYNAMCIS
        params = run_cavity( params )



    plt.plot( params.TIME / 41.341, np.abs(params.R_t[:,0,-1] - params.R_t[:,1,-1]), "-", c="black", label='$R^\\mathrm{z}_\\mathrm{H-F}$' )
    if ( params.do_cavity == True ): plt.plot( params.TIME / 41.341, params.qc_t[:], "-", c="red", label='$q_\\mathrm{c}$' )
    plt.legend()
    plt.xlabel( "Time (fs)", fontsize=15 )
    plt.savefig( "%s/POSITION_TIME.png" % (params.DATA_DIR) )
    plt.clf()

    get_IR( params )
    if ( params.do_cavity == True ):
        get_Transmission_Sectrum( params )

