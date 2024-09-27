from pyscf import scf, gto, cc, grad
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce

from time import time

def kernel( atom_labels, R, method ):
    at_string = ''
    for at in range( len(R) ):
        at_string += "%s %1.6f %1.6f %1.6f; " % (atom_labels[at], R[at,0], R[at,1], R[at,2])
    at_string = at_string[:-2] # Remove last semicolon

    basis = 'sto3g' # 'sto3g', '321g', '6311**g++', 'ccpvdz'
    mol = gto.M(
        atom = at_string,
        unit = 'bohr',
        basis = basis,
        symmetry = True,
        max_memory = 29_000, # in MB
        verbose = 0
        )


    if ( method == "HF" ):
        myhf     = mol.RHF(max_cycle=500).run()
        E_HF     = myhf.e_tot
        dm_hf    = myhf.make_rdm1()
        DIP_HF   = scf.hf.dip_moment(mol=mol,dm=dm_hf,unit='au',verbose=0) 
        GRAD_hf  = grad.rhf.Gradients(myhf).kernel()
        #print( "E_HF =", E_HF )
        return E_HF, DIP_HF, GRAD_hf

    elif ( method == "LDA" ):
        mydft     = mol.RKS(max_cycle=500, xc="lda,vwn").run()
        E_DFT     = mydft.e_tot
        dm_DFT    = mydft.make_rdm1()
        DIP_DFT   = scf.hf.dip_moment(mol=mol,dm=dm_DFT,unit='au',verbose=0) 
        GRAD_DFT  = grad.rks.Gradients(mydft).kernel()
        return E_DFT, DIP_DFT, GRAD_DFT

    elif ( method == "PBE" ):
        mydft     = mol.RKS(max_cycle=500, xc="pbe,pbe").run()
        E_DFT     = mydft.e_tot
        dm_DFT    = mydft.make_rdm1()
        DIP_DFT   = scf.hf.dip_moment(mol=mol,dm=dm_DFT,unit='au',verbose=0) 
        GRAD_DFT  = grad.rks.Gradients(mydft).kernel()
        return E_DFT, DIP_DFT, GRAD_DFT

    elif( method == "CCSD" ):
        myhf         = mol.RHF(max_cycle=500).run()
        mycc         = cc.CCSD(myhf).run()
        E_CCSD       = mycc.e_tot
        rdm1         = mycc.make_rdm1()
        rdm1         = reduce(np.dot,(myhf.mo_coeff,rdm1,myhf.mo_coeff.T)) # back to atomic representation
        DIP_CCSD  = scf.hf.dip_moment(mol,rdm1,unit='au',verbose=0)
        GRAD_CCSD = grad.ccsd.Gradients(mycc).kernel()
        return E_CCSD, DIP_CCSD, GRAD_CCSD

    else:
        print("Error: Electronic structure method not recognized")
        exit()


def do_el_structure( atom_labels, R, el_structure_method ):
    return kernel( atom_labels, R, el_structure_method )

def do_dipole_gradient( atom_labels, R, el_structure_method, is_diatomic=False ):
    if ( is_diatomic == True ):
        return do_dipole_gradient_diatomic_z( atom_labels, R, el_structure_method )

    T0 = time()
    dR       = 0.0001
    NATOMS   = len(atom_labels)
    E_GRAD   = np.zeros( (NATOMS, 3) )
    DIP_GRAD = np.zeros( (NATOMS, 3, 3) )
    
    
    for at in range( NATOMS ):
        for xyz in range(3):
            R_plus  = R.copy()
            R_minus = R.copy()
            R_plus[at,xyz]  += dR
            R_minus[at,xyz] -= dR
            E_plus,  MU_plus,  _ = kernel( atom_labels, R_plus, el_structure_method )
            E_minus, MU_minus, _ = kernel( atom_labels, R_minus, el_structure_method )
            E_GRAD[at,xyz]       = (E_plus - E_minus) / 2 / dR
            DIP_GRAD[at,xyz]     = (MU_plus - MU_minus) / 2 / dR
    
    # E0, DIP0, GRAD0 = kernel( atom_labels, R, el_structure_method )
    # print("Energy Gradients:")
    # print( "Exact:\n", GRAD0 )
    # print( "Approx:\n", E_GRAD )

    # print("Dipole Gradients:")
    # print( DIP_GRAD )

    print( "DIP GRAD Time: %1.3f s" % ( time() - T0 ) )
    return DIP_GRAD

def do_dipole_gradient_diatomic_z( atom_labels, R, el_structure_method ):
    T0 = time()
    dR       = 0.0001
    NATOMS   = len(atom_labels)
    E_GRAD   = np.zeros( (NATOMS, 3) )
    DIP_GRAD = np.zeros( (NATOMS, 3, 3) )
    
    R_plus  = R.copy()
    R_minus = R.copy()
    R_plus[0,-1]  += dR
    R_minus[0,-1] -= dR
    E_plus,  MU_plus,  _ = kernel( atom_labels, R_plus, el_structure_method )
    E_minus, MU_minus, _ = kernel( atom_labels, R_minus, el_structure_method )
    E_GRAD[0,-1]       = (E_plus - E_minus) / 2 / dR
    DIP_GRAD[0,-1]     = (MU_plus - MU_minus) / 2 / dR
    
    E_GRAD[1,-1]       = -E_GRAD[0,-1]
    DIP_GRAD[1,-1]     = -DIP_GRAD[0,-1]

    # E0, DIP0, GRAD0 = kernel( atom_labels, atom_coords )
    # print("Energy Gradients:")
    # print( "Exact:\n", GRAD0 )
    # print( "Approx:\n", E_GRAD )

    # print("Dipole Gradients:")
    # print( DIP_GRAD )

    print( "DIP GRAD Time: %1.3f s" % ( time() - T0 ) )

    return DIP_GRAD


if ( __name__ == "__main__" ):
    atom_labels = ['H', 'F']
    atom_coords = np.array([[0.0, 0.0, 0.0],[ 0.0, 0.0, 2.0]])
    E, DIP, GRAD = do_el_structure( atom_labels, atom_coords )

    do_dipole_gradient( atom_labels, atom_coords )
    do_dipole_gradient_diatomic_z( atom_labels, atom_coords )




