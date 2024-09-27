import numpy as np
import subprocess as sp

class Parameters():
    def __init__(self):
        self.NSTEPS = 10000 # Steps for nuclear propagation
        self.dtI_fs = 0.1 # fs

        self.el_structure_method = "HF" # "HF", "LDA", "PBE", "CCSD"

        # DEFINE INITIAL POSTIONS AND VELOCITIES
        self.atom_labels = ['H', 'F']
        self.R0 = np.array([[0.0, 0.0, 1.5],[ 0.0, 0.0, 0.0]]) # a.u.
        self.V0 = np.array([[0.0, 0.0, 0.0],[ 0.0, 0.0, 0.0]]) # a.u.

        # CAVITY PARAMETERS
        self.do_cavity           = True
        self.wc                  = 4100 # cm-1
        self.A0                  = 0.1 # a.u.
        self.qc0                 = 1.0 # a.u.
        self.pc0                 = 0.0 # a.u.
        self.cavity_polarization = np.array([0.0, 0.0, 1.0])

        self.build()

    def build(self):
        self.NATOMS  = len( self.atom_labels )
        self.dtI     = self.dtI_fs * 41.341 # a.u.
        self.TIME    = np.linspace(0.0, self.NSTEPS*self.dtI, self.NSTEPS) # a.u.
        self.MASSES  = get_masses( self.atom_labels ) # a.u.
        self.R_t     = np.zeros( (self.NSTEPS, len(self.atom_labels), 3) )
        self.V_t     = np.zeros( (self.NSTEPS, len(self.atom_labels), 3) )
        self.R_t[0]  = self.R0
        self.V_t[0]  = self.V0
        self.MU_t    = np.zeros( (self.NSTEPS, 3) )


        self.E_t     = np.zeros( (self.NSTEPS,2) ) # T_N, V_NUC
        self.DATA_DIR = "RESULTS_dt_%1.3f/" % ( self.dtI/41.341 )


        if ( self.do_cavity == True ):
            self.E_t     = np.zeros( (self.NSTEPS,5) ) # T_N, V_NUC, T_PH, V_PH, V_NUC-PH # TODO -- Add DSE
            self.qc_t    = np.zeros( (self.NSTEPS) )
            self.pc_t    = np.zeros( (self.NSTEPS) )
            self.qc_t[0] = self.qc0
            self.pc_t[0] = self.pc0
            self.DATA_DIR = "RESULTS_doCavity_%s_WC_%1.3f_A0_%1.3f_dt_%1.3f/" % ( str(self.do_cavity), self.wc, self.A0, self.dtI/41.341 )

            self.wc_cm  = self.wc * 1.0 # This is the input units
            self.wc_meV = self.wc_cm / 8.065
            self.wc     = self.wc_meV / 27.2114 / 1000


        assert( len(self.R0) == len(self.atom_labels) ), "Error: Number of atoms and atom labels do not match"
        if ( len(self.atom_labels) == 2 ):
            self.is_diatomic = True
        else:
            self.is_diatomic = False

        sp.call("mkdir -p %s" % self.DATA_DIR, shell=True)



def get_masses( atom_labels ):
    masses = []
    amu_to_au  = 1836/1.0079
    for at in atom_labels:
        if ( at == 'H' ):
            masses.append( 1.0071 )
        elif ( at == 'C' ):
            masses.append( 12.011 )
        elif ( at == 'N' ):
            masses.append( 14.007 )
        elif ( at == 'O' ):
            masses.append( 15.999 )
        elif ( at == 'F' ):
            masses.append( 18.998 )
        else:
            print("Error: Mass not entered for atom %s" % at)
            exit()
    return np.array( masses ) * amu_to_au


if ( __name__ == "__main__" ):
    params = Parameters()
    print( params.MASSES )