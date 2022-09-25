
#Parsec, Solar Mass, Megayear

_MSUN   = 1.98855e30 #kg
_PC     = 3.08567758149137e16 #m
_YR     = 365.25 * 24 * 3600 #s
_MYR    = 1e6*_YR

G_N     = 6.67408e-11/(_PC**3 * _MYR**-2 * _MSUN**-1) # m^3 s^-2 kg^-1 -> pc^3 Myr^-2 Msun^-1  
C       = 299792458.0/(_PC/_MYR) #m/s -> pc/Myr
