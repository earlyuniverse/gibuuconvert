#!/usr/bin/env python3

import numpy as np

__doc__ ="""
%prog FinalEvents.dat [options]
"""




#columns in file should be configured as below, where x denotes a position vector and p a momentum vector
#column heading 0=run number 1=event number  2=particle id 3=charge 4=perturbative weight 5=x[1] 6=x[2] 7=x[3] 8=p[0] 9=p[1] 10=p[2] 11=p[3] 12=history 13=production id 14=neutrino energy


def translate(data):
    """
    function to translate GiBUU particle ID's to PDG ID's
    """
    pdg=[]
    for i in range(len(data[:,0])):
        #data[:,2] is column containing GiBUU ID's, data[:,3] contains the charge

        #leptons

        if data[:,2][i]==903:
            pdg.append(15)        #tau minus
        elif data[:,2][i]==901:
            pdg.append(11)        #e minus
        elif data[:,2][i]==902:
            pdg.append(13)         #mu minus
        elif data[:,2][i]==911:
            pdg.append(12)       #nu_e
        elif data[:,2][i]==-911:
            pdg.append(-12)      #nu_e bar
        elif data[:,2][i]==912:
            pdg.append(14)        #nu_mu
        elif data[:,2][i]==-912:
            pdg.append(-14)       #nu_mu bar
        elif data[:,2][i]==913:
            pdg.append(16)        #nu_tau
        elif data[:,2][i]==-913:
            pdg.append(-16)       #nu_tau bar

        #light baryons

        elif data[:,2][i]==1:
            if data[:,3][i]==0:
                pdg.append(2112)   #Neutron
            if data[:,3][i]==1:
                pdg.append(2212)   #Proton
        elif data[:,2][i]==-1:
            if data[:,3][i]==-1:
                pdg.append(-2212)   #antiproton
            if data[:,3][i]==0:
                pdg.append(-2112)   #antineutron

        elif data[:,2][i]==2:
            if data[:,3][i]==2:
                pdg.append(2224)    #delta ++
            if data[:,3][i]==1:
                pdg.append(2214)    #delta +
            if data[:,3][i]==0:
                pdg.append(2114)    #delta_0
            if data[:,3][i]==-1:
                pdg.append(1114)    #delta -

        #light mesons

        elif data[:,2][i]==101:
            if data[:,3][i]==-1:
                pdg.append(-211)    #pi minus
            if data[:,3][i]==0:
                pdg.append(111)     #pi_0
            if data[:,3][i]==1:
                pdg.append(211)     #pi plus

        #charmed mesons

        elif data[:,2][i]==114:
            if data[:,3][i]==1:
                pdg.append(411)     #D plus
            if data[:,3][i]==0:
                pdg.append(421)     #D_0
        elif data[:,2][i]==115:
            if data[:,3][i]==-1:
                pdg.append(-411)    #D minus
            if data[:,3][i]==0:
                pdg.append(-421)    #D_0 BAR
        elif data[:,2][i]==118 and data[:,3][i]==1:
            pdg.append(431)                          #d sub s plus
        elif data[:,2][i]==119 and data[:,3][i]==-1:
            pdg.append(-431)      #d sub s minus

        #strange mesons

        elif data[:,2][i]==110:
            if data[:,3][i]==0:
                pdg.append(311)     #K_0
            if data[:,3][i]==1:
                pdg.append(321)     #K plus
        elif data[:,2][i]==111:
            if data[:,3][i]==-1:
                pdg.append(-321)    #K minus
            if data[:,3][i]==0:
                pdg.append(-311)    #K_0 bar

        #stange baryons

        elif data[:,2][i]==32 and data[:,3][i]==0:
            pdg.append(3122)        #lamda

        elif data[:,2][i]==53:
            if data[:,3][i]==0:
                pdg.append(3322)    #Xi_0
            if data[:,3][i]==-1:
                pdg.append(3312)    #Xi minus

        elif data[:,2][i]==33:
            if data[:,3][i]==1:
                pdg.append(3222)    #sigma plus
            if data[:,3][i]==0:
                pdg.append(3212)    #sigma 0
            if data[:,3][i]==-1:
                pdg.append(3112)    #sigma minus
        elif data[:,2][i]==-33:
            if data[:,3][i]==-1:
                pdg.append(-3222)   #sigma plus bar
            if data[:,3][i]==0:
                pdg.append(-3212)   #sigma 0 bar
            if data[:,3][i]==1:
                pdg.append(-3112)   #sigma minus bar

        elif data[:,2][i]==-32 and data[:,3][i]==0:
            pdg.append(-3122)     #lamda bar

        elif data[:,2][i]==-53:
            if data[:,3][i]==0:
                pdg.append(-3322)    #Xi_0 bar
            if data[:,3][i]==1:
                pdg.append(-3312)    #Xi minus bar

        elif data[:,2][i]==55 and data[:,3][i]==-1:
            pdg.append(3334)                          #ohm

        #charmed baryons

        elif data[:,2][i]==56 and data[:,3][i]==1:
            pdg.append(4122)        #Lamda sub c
        elif data[:,2][i]==-56 and data[:,3][i]==-1:
            pdg.append(-4122)                         #lamda-sub-c bar
        elif data[:,2][i]==57:
            if data[:,3][i]==2:
                pdg.append(4222)     #sigma sub c plus plus
            if data[:,3][i]==1:
                pdg.append(4212)     #sigma sub c plus
            if data[:,3][i]==0:
                pdg.append(4112)     #sigma sub c 0
        elif data[:,2][i]==59:
            if data[:,3][i]==0:
                pdg.append(4132)    #Xi sub c 0
            if data[:,3][i]==1:
                pdg.append(4232)    #Xi sub c plus

        else:
            print(data[:,2][i])
            print(data[:,3][i])
            print('undefined particle type')
    return pdg

def find_cs(data):
    """
    finds the cross section of the simulated interactions by summing the perturbative weights
    """
    if len(data[:,0])==0: #incase datafile is empty but need a return for later use
        return []
    else:
        nu=[]
        for i in range(len(data[:,0])-1):
            #perturbative weight is the same for every
            #particle within a given event except the collision neutron, whose pw=0
            #function extracts one non zero perturbative weight from each event
            if data[:,1][i+1]>data[:,1][i] and data[:,4][i+1]!=0:
                nu.append(data[:,4][i+1])
            if data[:,1][i+1]>data[:,1][i] and data[:,4][i+1]==0:  #ensuring 0 perturbative weight interaction neutrons are skipped
                nu.append(data[:,4][i+2])
    c_s=sum(nu)/data[:,0][len(data[:,0])-1]  #(sum of perturbative weight of each event)/(number of runs)
    return c_s

def radius(data):
    """
    finding magnitude of the position 3 vector
    """
    rad=[]
    for i in range(len(data[:,0])):
        rad.append(np.sqrt(((data[:,5][i])**2)+((data[:,6][i])**2)+((data[:,7][i])**2)))
    return rad

def filter_perweight(data):
    """
    removing collision neutron data from event file
    """
    j=[]
    for i in range(len(data[:,0])):
        if data[:,4][i]<=0:
            j.append(i)
    new_data=np.delete(data,j,0)
    return new_data

def filter_tau(data):
    """
    locating taus in event file and isolating their data
    """
    l=[]
    for i in range(len(data[:,0])):
        if data[:,2][i]== 903:
            continue
        else:
            l.append(i)
    lep=np.delete(data,l,0)
    return lep

def filter_particles(data):
    """
    removing taus from datafile
    """
    j=[]
    for i in range(len(data[:,0])):
        if data[:,2][i]!=903:
            continue
        else:
            j.append(i)
    par=np.delete(data,j,0)
    return par

def comb_re(data):
    """
    combining information in run and event columns for final dataframe
    """
    t=[]
    for i in range(len(data[:,0])):
        x=int(data[:,0][i])
        y=int(data[:,1][i])
        z='"'+str(x)+'_'+str(y)+'"'

        t.append(z)
    return t


#DECAYING pi_0 to two photons --- this is a verbatim translation of the corresponding fortran routines in tauola

def calcR(p1,p2,p3,e):
    """
    finding the rest mass of the pion to be decayed
    Credit to TAUOLA
    """
    return np.sqrt(e**2 - p1**2 - p2**2 - p3**2)/2      #mass of pi/2

def spherd(R):
    """
    gives four momentum of photon
    Credit to TAUOLA
    """
    r1, r2 = np.random.rand(2)
    costh = 2*r1 -1
    sinth = np.sqrt(1 - costh**2)
    return [R*sinth*np.cos(2*np.pi*r2), R*sinth*np.sin(2*np.pi*r2), R*costh, R]  #doesn't incl metric

def boost(v, p):
    """
    v is momentum 4 vector pion and p is four momentum of photons
    Credit to TAUOLA
    """
    amv = np.sqrt(abs(v[3]**2 - v[0]**2 -v[1]**2 -v[2]**2))
    t = (p[0]*v[0] + p[1]*v[1] + p[2]*v[2] + p[3]*v[3])/amv
    wsp = (t + p[3])/(v[3] + amv)     #
    return [p[0] + wsp*v[0], p[1] + wsp*v[1], p[2] + wsp*v[2], t]

def decay(p1, p2, p3, e):
    """
    decays pion to two photons in its rest frame then boosts back to the lab frame and returns photon kinematics
    Credit to TAUOLA
    """
    R = calcR(p1, p2, p3, e)
    X = spherd(R)
    Y = [-X[0], -X[1], -X[2], R]

    phot1 = boost( [p1,p2,p3,e], X)
    phot2 = boost( [p1,p2,p3,e], Y)
    return phot1, phot2

def decayfile(inputfile, outputfile, decaythese=["111"]):
    """
    Search for pions in datafile and decays them as above
    Output new file with pions replaced by their decay products(two photons)
    """

    g = open(outputfile, "w")
    with open(inputfile) as f:
        comments=f.readline()
        g.write(comments)
        for line in f:
            l = line.split(",")
            pid = l[1]
            if pid in decaythese:
                p1 = float(l[3])
                p2 = float(l[4])
                p3 = float(l[5])
                e  = float(l[2])
                PH1, PH2 = decay(p1,p2,p3,e)
                l1 = [x for x in l]
                l2 = [x for x in l]
                l1[1] = "22"
                l1[2] = str(PH1[-1])
                l1[3] = str(PH1[0])
                l1[4] = str(PH1[1])
                l1[5] = str(PH1[2])
                l2[1] = "22"
                l2[2] = str(PH2[-1])
                l2[3] = str(PH2[0])
                l2[4] = str(PH2[1])
                l2[5] = str(PH2[2])
                g.write(",".join(l1))
                g.write(",".join(l2))
            else:
                g.write(line)
    g.close()

if __name__ == "__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-f", "--force", dest="FORCE", action="store_true", default=False, help="Allow overwriting")
    op.add_option("-p", dest="PARTICLES",    default="particles.csv",        help="Output filename for undecayed particles (except tau) (default: %default)")
    op.add_option("-t", dest="TAUS",         default="taus.csv",             help="Output filename for undecayed taus) (default: %default)")
    op.add_option("-d", dest="DECPARTICLES", default="decayedparticles.csv", help="Output filename for decayed particles (except taus)) (default: %default)")
    op.add_option("-s", dest="SEED", default=7627834, type=int, help="Random seed (for pi0 decays) (default: %default)")
    opts, args = op.parse_args()

    np.random.seed(opts.SEED)

    if len(args)!=1:
        print("Need exactly one input file, user provided {}, exiting...".format(len(args)))
        sys.exit(1)

    inputgibuu = args[0]

    if not os.path.exists(inputgibuu):
        print("Error, input file {} not found, exiting...",format(inputgibuu))
        sys.exit(1)

    # Some protections
    if inputgibuu in [opts.PARTICLES, opts.TAUS, opts.DECPARTICLES]:
        print("Error, one of the output filenames is identical to the input file, exiting...")
        sys.exit(1)

    if (opts.PARTICLES == opts.TAUS) or (opts.PARTICLES == opts.DECPARTICLES) or (opts.TAUS == opts.DECPARTICLES):
        print("Error, output file names must be distinct, exiting...")
        sys.exit(1)

    if not opts.FORCE:
        for fout in [opts.PARTICLES, opts.TAUS, opts.DECPARTICLES]:
            if os.path.exists(fout):
                print("Error, output file {} exists and overwriting is disabled by default (use -f to enable) exiting...".format(fout))
                sys.exit(1)


    # loading data from specified file, this file will be converted to 2 csv
    # files, one filled with all particles but taus, with any pi_0s decayed to
    # photons, (decayedparticles.csv) and the other filled with all tau leptons
    # which can then be decayed using tauola (taus.csv)
    DAT=np.loadtxt(inputgibuu)

    xs_p=find_cs(DAT)
    print("Cross-section: {} 1e-38 cm^2".format(xs_p))

    filtered=filter_perweight(DAT)  #removing per_weight=0 neutrons

    PARS=filter_particles(filtered)
    PARS_mod=comb_re(PARS)             #reformatting run and event information
    TAUS=filter_tau(filtered)        #making array only including taus
    TAUS_mod=comb_re(TAUS)

    #translating particle ID's
    r_p=translate(PARS)
    r_l=translate(TAUS)

    #making array of cross section to fill columns in dataframe with
    xsc_p=np.linspace(xs_p,xs_p,len(PARS[:,0]))


    import pandas
    import csv

    #making dataframes with column headings of: run_event, pdgid, p_0, px, py, pz, enu, perweight, xs
    #for all particles but tau
    out_particles={'"run_event"':PARS_mod,'"pdgid"':r_p,'"E"': PARS[:,8],'"px"':PARS[:,9],'"py"':PARS[:,10],'"pz"':PARS[:,11],'"Enu"': PARS[:,14],'"weight"':PARS[:,4],'"xsec"':xsc_p}
    out_blank={'"run_event"':[],'"pdgid"':[],'"E"': [],'"px"':[],'"py"':[],'"pz"':[],'"Enu"': [],'"weight"':[],'"xsec"':[]}
    #taus
    out_tau={'"run_event"': TAUS_mod,'"Enu"':TAUS[:,14],'"E"':TAUS[:,8],'"px"':TAUS[:,9],'"py"':TAUS[:,10],'"pz"':TAUS[:,11]}
    df=pandas.DataFrame(data=out_particles) #dataframe of all particles excl taus
    df_l=pandas.DataFrame(data=out_tau)     #dataframe of taus
    df_o=pandas.DataFrame(data=out_blank)   ##empty dataframe to store particle file after pion decay

    #converting to CSV
    df.to_csv  (opts.PARTICLES   , index = False,quoting=csv.QUOTE_NONE) # making csv of non taus BEFORE pi_0 decay, this is only used as an input to the decay function
    df_o.to_csv(opts.DECPARTICLES, index = False,quoting=csv.QUOTE_NONE) # making empty csv to store particles after decay
    df_l.to_csv(opts.TAUS        , index = False,quoting=csv.QUOTE_NONE) # making csv of taus


    # decaying pions in file particles.csv and putting decayed data into decayedparticles.csv
    decayfile(opts.PARTICLES, opts.DECPARTICLES)
    print("Undecayed taus are written to {}. Process these with TAUOLA".format(opts.TAUS))
    print("All other gibuu produced particels are written to {}.".format(opts.PARTICLES))
    print("A copy of the same information but with all pi0s decayed to photons are written to {}.".format(opts.DECPARTICLES))
    print("Done.")
    # NOTE this function takes an optional third argument which curently only holds pi0 (111)--- trivial to extend to other pids:
    # decayfile(opts.PARTICLES, opts.DECPARTICLES, decaythese=["111", "221"])
