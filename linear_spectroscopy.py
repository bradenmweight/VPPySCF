import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from emcee.autocorr import function_1d as get_ACF

def do_FFT( p, ACF ):
    dt    = p.TIME[1] - p.TIME[0] # a.u.
    ACF_w = np.fft.fft( ACF )[1:p.NSTEPS//2].real
    w_au  = np.fft.fftfreq( len(ACF) )[1:p.NSTEPS//2] * 2 * np.pi / dt 
    w_cm  = w_au * 27.2114 * 1000 * 8.065
    w_meV = w_au * 27.2114 * 1000
    #print( "MIN, MAX (cm-1)", w_cm[0], w_cm[-1] )
    #print( "MIN, MAX (meV)", w_meV[0], w_meV[-1] )
    return ACF_w, w_cm, w_meV

def get_IR( p ):
    """
    MU: Dipole Moment Time Series
    TIME: Time Series (fs)
    """

    # PLOT DIPOLE TIME SERIES
    plt.plot( p.TIME / 41.341, p.MU_t[:,-1] ) # Z-projection
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("Dipole (a.u.)", fontsize=15)
    plt.xlim( 0 )
    plt.tight_layout()
    plt.savefig("%s/DIPOLE_TIME.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

    # GET AUTO-CORRELATION FUNCTION OF DIPOLE
    ACF = get_ACF( p.MU_t[:,-1] ) # Z-projection
    plt.plot( p.TIME / 41.341, ACF )
    plt.xlabel("Lag Time (fs)", fontsize=15)
    plt.ylabel("Auto-correlation Function", fontsize=15)
    #plt.xlim( 0, 2000 )
    plt.xlim( 0 )
    plt.tight_layout()
    plt.savefig("%s/DIPOLE_ACF.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

    # FFT[ACF]
    ACF_w, w_cm, w_meV = do_FFT( p, ACF )

    # Save cm^-1
    plt.plot( w_cm, ACF_w, "-", c='black' )
    plt.xlim(3500, 5500)
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
    plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/IR_SPEC_cm-1.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

    # Save meV
    plt.plot( w_meV, ACF_w, "-", c='black' )
    #plt.xlim(w_meV[0],w_meV[-1])
    plt.xlabel("Energy (meV)", fontsize=15)
    plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/IR_SPEC_meV.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

def get_Transmission_Sectrum( p ):
    """
    QC: Photon Coordinate Time Series
    TIME: Time Series (fs)
    """

    # PLOT DIPOLE TIME SERIES
    plt.plot( p.TIME / 41.341, p.qc_t[:] )
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("QC (a.u.)", fontsize=15)
    plt.xlim( 0 )
    plt.tight_layout()
    plt.savefig("%s/QC_TIME.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

    # GET AUTO-CORRELATION FUNCTION OF QC
    ACF = get_ACF( p.qc_t[:] )
    plt.plot( p.TIME / 41.341, ACF )
    plt.xlabel("Lag Time (fs)", fontsize=15)
    plt.ylabel("Auto-correlation Function", fontsize=15)
    #plt.xlim( 0, 2000 )
    plt.xlim( 0 )
    plt.tight_layout()
    plt.savefig("%s/QC_ACF.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()


    # FFT[ACF]
    ACF_w, w_cm, w_meV = do_FFT( p, ACF )

    print( "Expected Cavity Frequency (a.u., cm^-1, meV)", p.wc, p.wc * 27.2114 * 1000 * 8.065, p.wc * 27.2114 * 1000 )
    print( "Expected Cavity Period (a.u., fs)", 2*np.pi / p.wc, 2*np.pi / p.wc / 41.341 )

    # Save cm^-1
    plt.plot( w_cm, ACF_w, "-", c='black' )
    plt.xlim(3500, 5500)
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
    plt.ylabel("Transmission (Arb. Units)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/TM_SPEC_cm-1.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()

    plt.plot( w_meV, ACF_w, "-", c='black' )
    # Save meV
    #plt.xlim(w_meV[0],w_meV[-1])
    plt.xlabel("Energy (meV)", fontsize=15)
    plt.ylabel("Transmission (Arb. Units)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/TM_SPEC_meV.jpg" % (p.DATA_DIR), dpi=300)
    plt.clf()


if ( __name__ == "__main__" ):
    pass
