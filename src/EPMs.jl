@doc """Empirical pseudopotential models for testing band energy calculation methods.
Models based on pseudopotential derivation explained in the textbook Solid
State Physics by Grosso and Parravicini.

Pseudopotential form factors taken from The Fitting of Pseudopotentials to
Experimental Data by Cohen and Heine.

Lattice constants from https://periodictable.com.
"""
module EPMs

using SymmetryReduceBZ.Lattices: genlat_FCC, genlat_BCC, genlat_HEX,
    genlat_BCT, get_recip_latvecs
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using PyCall: pyimport
using PyPlot: subplots
using QHull: chull,Chull
using SymmetryReduceBZ.Lattices: get_recip_latvecs
using SymmetryReduceBZ.Utilities: sample_circle, sample_sphere
using LinearAlgebra: norm, Symmetric, eigvals, dot
using SparseArrays: SparseMatrixCSC
using Distances: SqEuclidean, pairwise!
using Arpack: eigs



export eval_epm, RytoeV, epm₋model2D, epm₋model

Ag_name = "Ag"
Al_name = "Al"
Au_name = "Au"
Cs_name = "Cs"
Cu_name = "Cu"
In_name = "In"
K_name = "K"
Li_name = "Li"
Na_name = "Na"
Pb_name = "Pb"
Rb_name = "Rb"
Sn_name = "Sn"
Zn_name = "Zn"

epm_names = [Ag_name,Al_name,Au_name,Cs_name,Cu_name,In_name,K_name,Li_name,
    Na_name,Pb_name,Rb_name,Sn_name,Zn_name]

# The lattice types of the EPMs (follows the naming convention
# of High-throughput electronic band structure calculations:
# Challenges and tools by Wahyu Setyawan and Stefano Curtarolo).
Ag_type = "FCC"
Al_type = "FCC"
Au_type = "FCC"
Cs_type = "BCC"
Cu_type = "FCC"
In_type = "BCT₂"
K_type = "BCC"
Li_type = "BCC"
Na_type = "BCC"
Pb_type = "FCC"
Rb_type = "BCC"
Sn_type = "BCT₁"
Zn_type = "HEX"

# The lattice type of the reciprocal lattice
Ag_rtype = "BCC"
Al_rtype = "BCC"
Au_rtype = "BCC"
Cs_rtype = "FCC"
Cu_rtype = "BCC"
In_rtype = "BCT₂" # This is the real lattice type, reciprocal lattice type?
K_rtype = "FCC"
Li_rtype = "FCC"
Na_rtype = "FCC"
Pb_rtype = "BCC"
Rb_rtype = "FCC"
Sn_rtype = "BCT₁" # This is the real lattice type, reciprocal lattice type?
Zn_rtype = "HEX"

#= 
Symmetry preserving offset for each reciprocal lattice type in terms of 
fractions of the grid generating vectors. These were taken from a table from 
"Generalized regular k-point grid generation on the fly" by Wiley Morgan et al.
or resulted from tests for the given models since the reciprocal lattice of a 
body-centered tetragonal lattice may be different lattice types depending on the
lattice parameters (I think). 
=#
sym_offset = Dict("BCC" => [0,0,0],"FCC" => [0.5,0.5,0.5],
    "HEX" => [0,0,0.5], "BCT₁" => [0.5,0.5,0.5], "BCT₂" => [0.5,0.5,0.5],
    "square" => [0.5,0.5], "hexagonal" => [0.0,0.0], "centered rectangular" => 
    [0.0,0.0], "rectangular" => [0.0, 0.0], "oblique" => [0.0,0.0])

# The lattice angles of the EPMs in radians
Ag_αβγ = [π/2, π/2, π/2]
Al_αβγ = [π/2, π/2, π/2]
Au_αβγ = [π/2, π/2, π/2]
Cs_αβγ = [π/2, π/2, π/2]
Cu_αβγ = [π/2, π/2, π/2]
In_αβγ = [π/2, π/2, π/2]
K_αβγ = [π/2, π/2, π/2]
Li_αβγ = [π/2, π/2, π/2]
Na_αβγ = [π/2, π/2, π/2]
Pb_αβγ = [π/2, π/2, π/2]
Rb_αβγ = [π/2, π/2, π/2]
Sn_αβγ = [π/2, π/2, π/2]
Zn_αβγ = [π/2, π/2, 2π/3]

# The lattice constants of the EPMs in Bohr radii.
Ag_abc = [7.7201, 7.7201, 7.7201]
Al_abc = [7.6524, 7.6524, 7.6524]
Au_abc = [7.7067, 7.7067, 7.7067]
Cs_abc = [11.6048, 11.6048, 11.6048]
Cu_abc = [6.8312, 6.8312, 6.8312]
In_abc = [6.1460, 6.1460, 9.3468]
K_abc = [10.0685, 10.0685, 10.0685]
Li_abc = [6.6329, 6.6329, 6.6329]
Na_abc = [8.1081, 8.1081, 8.1081]
Pb_abc = [9.3557, 9.3557, 9.3557]
Rb_abc = [10.5541, 10.5541, 10.5541]
Sn_abc = [11.0205, 11.0205, 6.0129]
Zn_abc = [5.0359, 5.0359, 9.3481]

# The primitive lattice vectors of the EPMs
Ag_latvecs = genlat_FCC(Ag_abc[1])
Al_latvecs = genlat_FCC(Al_abc[1])
Au_latvecs = genlat_FCC(Au_abc[1])
Cs_latvecs = genlat_BCC(Cs_abc[1])
Cu_latvecs = genlat_FCC(Cu_abc[1])
In_latvecs = genlat_BCT(In_abc[1],In_abc[3])
K_latvecs = genlat_BCC(K_abc[1])
Li_latvecs = genlat_BCC(Li_abc[1])
Na_latvecs = genlat_BCC(Na_abc[1])
Pb_latvecs = genlat_FCC(Pb_abc[1])
Rb_latvecs = genlat_BCC(Rb_abc[1])
Sn_latvecs = genlat_BCT(Sn_abc[1],Sn_abc[3])
Zn_latvecs = genlat_HEX(Zn_abc[1],Zn_abc[3])

# Reciprocal lattice vectors
Ag_rlatvecs = get_recip_latvecs(Ag_latvecs,"angular")
Al_rlatvecs = get_recip_latvecs(Al_latvecs,"angular")
Au_rlatvecs = get_recip_latvecs(Au_latvecs,"angular")
Cs_rlatvecs = get_recip_latvecs(Cs_latvecs,"angular")
Cu_rlatvecs = get_recip_latvecs(Cu_latvecs,"angular")
In_rlatvecs = get_recip_latvecs(In_latvecs,"angular")
K_rlatvecs = get_recip_latvecs(K_latvecs,"angular")
Li_rlatvecs = get_recip_latvecs(Li_latvecs,"angular")
Na_rlatvecs = get_recip_latvecs(Na_latvecs,"angular")
Pb_rlatvecs = get_recip_latvecs(Pb_latvecs,"angular")
Rb_rlatvecs = get_recip_latvecs(Rb_latvecs,"angular")
Sn_rlatvecs = get_recip_latvecs(Sn_latvecs,"angular")
Zn_rlatvecs = get_recip_latvecs(Zn_latvecs,"angular")

# EPM rules for replacing distances with pseudopotential form factors
# Distances are for the angular reciprocal space convention.
Ag_dist_ff = [[1.99,7.95],[0.195,0.121]]
Al_dist_ff = [[2.02,8.09],[0.0179,0.0562]]
Au_dist_ff = [[1.99,7.98],[0.252,0.152]]
Cs_dist_ff = [[1.76],[-0.03]]
Cu_dist_ff = [[6.77,10.15],[0.282,0.18]]
In_dist_ff = [[8.36,10.17],[0.02,-0.047]]
K_dist_ff = [[2.34,3.12],[0.0075,-0.009]]
Li_dist_ff = [[5.38],[0.11]]
Na_dist_ff = [[3.60],[0.0158]]
Pb_dist_ff = [[1.35,5.41],[-0.084,-0.039]]
Rb_dist_ff = [[2.13],[-0.002]]
Sn_dist_ff = [[2.72,5.67,14.05,20.07],[-0.056,-0.069,0.051,0.033]]
Zn_dist_ff = [[1.81,2.08,2.53],[-0.022,0.02,0.063]]

Ag_rules = [1.99 => 0.195,0.195 => 0.121]
Al_rules = [2.02 => 0.0179,8.09 => 0.0562]
Au_rules = [1.99 => 0.252,7.98 => 0.152]
Cs_rules = [1.76 => -0.03]
Cu_rules = [3.19 => 0.18,6.77 => 0.282]
In_rules = [2.89 => 0.02,3.19 => -0.047]
K_rules = [1.77 => -0.009,1.53 => 0.0075]
Li_rules = [2.32 => 0.11]
Na_rules = [1.9 => 0.0158]
Pb_rules = [2.33 => -0.039,1.16 => -0.084]
Rb_rules = [1.46 => -0.002]
Sn_rules = [4.48 => 0.033,1.65 => -0.056,2.38 => -0.069,3.75 => 0.051]
Zn_rules = [1.34 => -0.022,1.59 => 0.063,1.44 => 0.02]

Ag_electrons = 1
Al_electrons = 3
Au_electrons = 1
Cs_electrons = 1
Cu_electrons = 1
In_electrons = 3
K_electrons = 1
Li_electrons = 1
Na_electrons = 1
Pb_electrons = 4
Rb_electrons = 1
Sn_electrons = 4
Zn_electrons = 2

Ag_sheets = 1
Al_sheets = 4
Au_sheets = 1
Cs_sheets = 2
Cu_sheets = 1
In_sheets = 4 
K_sheets = 2
Li_sheets = 1
Na_sheets = 2
Pb_sheets = 4
Rb_sheets = 2 
Sn_sheets = 5 
Zn_sheets = 3 

# Cutoffs are chosen so that there are at least 1000 terms in the expansion at
# the origin.
Ag_cutoff = 8.1
Al_cutoff = 8.2
Au_cutoff = 8.2
Cs_cutoff = 4.3
Cu_cutoff = 9.2
In_cutoff = 7.1
K_cutoff = 5.0
Li_cutoff = 7.5
Na_cutoff = 6.2
Pb_cutoff = 6.7
Rb_cutoff = 4.7
Sn_cutoff = 5.5
Zn_cutoff = 6.6

#=
Cutoffs are chosen such that the mean deviation in all eigenvalues is less
than 1e-10 for all points of a sparse mesh over the IBZ for all eigenvalues 
beneath and slightly above the Fermi level (up to 1 eV above). The mean 
deviation is for 5 different consecutive expansions where the number of terms in
the expansions for all k-points changed.
=#
Ag_cutoff = 10.47 # 2232 terms, 3 seconds
Al_cutoff = 10.49 # 2186 terms, 3 seconds 
Au_cutoff = 10.48 # 2230 terms, 3 seconds
Cs_cutoff = 5.86 # 2657 terms, 5 seconds
Cu_cutoff = 12.48 # 2617 terms, 5 seconds
In_cutoff = 9.79 # 2797  terms, 5 seconds
K_cutoff = 6.06 # 1916 terms, 2 seconds
Li_cutoff = 10.85 # 3144 terms, 7 seconds
Na_cutoff = 7.40 # 1878 terms, 2 seconds
Pb_cutoff = 8.58 # 2201 terms, 3 seconds
Rb_cutoff = 4.58 # 976 terms, 1 second
Sn_cutoff = 0
Zn_cutoff = 7.59 # 1498 terms, 0.5 seconds

# Ag_cutoff = 8.1
# Al_cutoff = 7.12
# Au_cutoff = 8.48
# Cs_cutoff = 4.4
# Cu_cutoff = 10.21
# In_cutoff = 4.27
# K_cutoff = 4.05
# Li_cutoff = 7.7
# Na_cutoff = 4.78
# Pb_cutoff = 5.82
# Rb_cutoff = 3.04
# Sn_cutoff = 6.92
# Zn_cutoff = 4.67

eVtoRy = 0.07349864435130871395
RytoeV = 13.6056931229942343775

Ag_bz = chull([0.0 -0.8138735647439266 -0.4069367823719631; -2.25895292678115e-17 -0.8138735647439265 0.4069367823719631; 0.4069367823719631 -0.8138735647439266 0.0; -0.40693678237196307 -0.8138735647439265 0.0; -6.776858780343452e-17 0.4069367823719631 0.8138735647439266; 0.4069367823719631 -4.5179058535623017e-17 0.8138735647439266; -0.4069367823719631 -4.517905853562301e-17 0.8138735647439265; -4.517905853562301e-17 -0.4069367823719631 0.8138735647439265; 0.8138735647439265 -0.40693678237196307 0.0; 0.8138735647439265 0.0 -0.40693678237196307; 0.8138735647439265 -2.2589529267811496e-17 0.40693678237196307; 0.8138735647439265 0.40693678237196307 0.0; -6.776858780343451e-17 0.8138735647439265 0.40693678237196307; -4.517905853562301e-17 0.8138735647439265 -0.40693678237196307; -0.4069367823719631 0.8138735647439265 0.0; 0.4069367823719631 0.8138735647439266 0.0; 0.0 -0.40693678237196307 -0.8138735647439265; -0.40693678237196307 0.0 -0.8138735647439265; 0.4069367823719631 0.0 -0.8138735647439266; -2.25895292678115e-17 0.4069367823719631 -0.8138735647439266; -0.8138735647439265 0.4069367823719631 0.0; -0.8138735647439265 -2.25895292678115e-17 0.4069367823719631; -0.8138735647439266 0.0 -0.4069367823719631; -0.8138735647439266 -0.4069367823719631 0.0])
Ag_ibz = chull([0.0 0.0 0.0; 0.4069367823719631 -0.4069367823719631 -0.40693678237196307; 0.0 -0.6104051735579449 -0.6104051735579448; 9.094148220876481e-17 -0.8138735647439265 0.0; 0.2034683911859815 -0.8138735647439264 -0.20346839118598137; 4.547074110438242e-17 -0.8138735647439265 -0.40693678237196307])

Al_bz = chull([-2.7043196271590465e-16 -0.8210738209162598 -0.410536910458129; 9.014398757196787e-17 -0.8210738209162597 0.41053691045812973; 0.410536910458129 -0.8210738209162598 4.440892098500629e-16; -0.41053691045812973 -0.8210738209162599 6.661338147750941e-16; 6.310079130037775e-16 0.41053691045812984 0.8210738209162597; 0.4105369104581301 4.527469748955415e-16 0.8210738209162596; -0.41053691045812896 2.7245899975160513e-16 0.8210738209162599; 4.507199378598411e-16 -0.41053691045812923 0.8210738209162599; 0.8210738209162596 -0.4105369104581292 2.2204460492503114e-16; 0.8210738209162598 3.2594311394716253e-16 -0.4105369104581295; 0.8210738209162598 0.41053691045812984 0.0; 0.8210738209162597 3.6158946880572304e-16 0.4105369104581297; 4.507199378598408e-16 0.8210738209162599 0.4105369104581294; 9.014398757196834e-17 0.8210738209162598 -0.4105369104581295; -0.410536910458129 0.82107382091626 2.2204460492503104e-16; 0.41053691045812973 0.8210738209162599 0.0; -2.704319627159046e-16 -0.41053691045812923 -0.8210738209162598; -0.4105369104581296 2.0116629003448417e-16 -0.8210738209162598; 0.4105369104581294 3.8145426517842045e-16 -0.82107382091626; -9.014398757196833e-17 0.4105369104581297 -0.8210738209162599; -0.8210738209162599 0.41053691045812896 4.44089209850063e-16; -0.8210738209162599 1.0135185178497033e-18 0.4105369104581294; -0.8210738209162597 -3.463283634071056e-17 -0.41053691045812885; -0.8210738209162599 -0.4105369104581296 6.661338147750938e-16])
Al_ibz = chull([8.326672684688695e-17 -0.6158053656871946 -0.6158053656871946; 0.41053691045812984 -0.41053691045812984 -0.41053691045812984; 0.0 0.0 0.0; 1.665334536937741e-16 -0.8210738209162598 1.1102230246251605e-16; 0.20526845522906484 -0.8210738209162596 -0.20526845522906464; 1.387778780781451e-16 -0.8210738209162599 -0.41053691045812923])

Au_bz = chull([-2.7235089737372103e-16 -0.8152886848040779 -0.4076443424020392; -4.552601897861454e-17 -0.8152886848040778 0.40764434240203845; 0.40764434240203873 -0.8152886848040779 -1.4846869033426806e-16; -0.40764434240203906 -0.8152886848040779 -2.2270303550140216e-16; 1.1314403369823112e-16 0.4076443424020385 0.8152886848040778; 0.4076443424020385 4.525761347929246e-17 0.8152886848040778; -0.4076443424020383 -9.091783520756786e-17 0.8152886848040777; 4.512341072963148e-17 -0.40764434240203845 0.8152886848040777; 0.8152886848040779 -0.40764434240203873 -7.423434516713401e-17; 0.8152886848040776 -1.4846869033426801e-16 -0.4076443424020386; 0.8152886848040777 2.262880673964622e-17 0.40764434240203873; 0.8152886848040777 0.40764434240203873 0.0; 9.051522695858489e-17 0.8152886848040777 0.4076443424020387; -1.3630965143652146e-16 0.8152886848040777 -0.4076443424020387; -0.4076443424020385 0.8152886848040778 -7.4234345167134e-17; 0.4076443424020387 0.8152886848040777 0.0; -4.0852634606058145e-16 -0.4076443424020393 -0.8152886848040777; -0.4076443424020393 -4.3311282935539664e-16 -0.8152886848040779; 0.40764434240203845 -2.9693738066853623e-16 -0.8152886848040778; -3.4050572309198175e-16 0.4076443424020385 -0.8152886848040778; -0.8152886848040779 0.40764434240203823 -1.4846869033426816e-16; -0.8152886848040778 -2.4972209063407477e-16 0.4076443424020382; -0.8152886848040777 -4.2081958770798897e-16 -0.40764434240203906; -0.8152886848040778 -0.40764434240203906 -2.227030355014021e-16])
Au_ibz = chull([0.0 0.0 0.0; 0.40764434240203873 -0.40764434240203873 -0.40764434240203923; 0.0 -0.6114665136030581 -0.6114665136030584; -4.085263460605815e-16 -0.8152886848040778 2.269590811447675e-17; 0.2038221712010191 -0.8152886848040778 -0.2038221712010196; -2.0426317303029082e-16 -0.8152886848040778 -0.4076443424020389])

Cs_bz = chull([1.1102230246251565e-16 5.551115123125783e-17 -0.5414298658468555; -0.27071493292342774 0.27071493292342774 -0.27071493292342774; 0.0 0.5414298658468555 0.0; 0.27071493292342774 -0.27071493292342774 0.27071493292342774; 0.5414298658468555 0.0 0.0; 0.27071493292342774 0.27071493292342774 0.27071493292342774; 0.27071493292342774 0.27071493292342774 -0.27071493292342774; 0.2707149329234278 -0.27071493292342774 -0.27071493292342774; -0.2707149329234277 0.27071493292342774 0.27071493292342774; 5.551115123125784e-17 5.551115123125784e-17 0.5414298658468554; -0.27071493292342774 -0.27071493292342774 0.2707149329234277; -0.2707149329234277 -0.27071493292342774 -0.2707149329234277; 5.551115123125783e-17 -0.5414298658468555 0.0; -0.5414298658468556 -2.775557561562892e-17 -2.775557561562892e-17])
Cs_ibz = chull([0.0 0.0 0.0; 0.0 0.2707149329234277 -0.27071493292342774; -0.2707149329234277 0.2707149329234277 -0.2707149329234278; 0.0 0.0 -0.5414298658468555])

Cu_bz = chull([0.0 -0.919777682863858 -0.4598888414319291; 0.0 -0.919777682863858 0.4598888414319291; 0.4598888414319291 -0.919777682863858 0.0; -0.4598888414319291 -0.919777682863858 0.0; 0.0 0.4598888414319291 0.919777682863858; 0.4598888414319291 0.0 0.919777682863858; -0.4598888414319291 0.0 0.919777682863858; 0.0 -0.4598888414319291 0.919777682863858; 0.919777682863858 -0.4598888414319291 0.0; 0.919777682863858 0.0 -0.4598888414319291; 0.919777682863858 0.4598888414319291 0.0; 0.919777682863858 0.0 0.4598888414319291; 0.0 0.919777682863858 0.4598888414319291; 0.0 0.919777682863858 -0.4598888414319291; -0.4598888414319291 0.919777682863858 0.0; 0.4598888414319291 0.919777682863858 0.0; 0.0 -0.4598888414319291 -0.919777682863858; -0.4598888414319291 0.0 -0.919777682863858; 0.4598888414319291 0.0 -0.919777682863858; 0.0 0.4598888414319291 -0.919777682863858; -0.919777682863858 0.4598888414319291 0.0; -0.919777682863858 0.0 0.4598888414319291; -0.919777682863858 0.0 -0.4598888414319291; -0.919777682863858 -0.4598888414319291 0.0])
Cu_ibz = chull([0.0 -0.6898332621478935 -0.6898332621478935; 0.459888841431929 -0.459888841431929 -0.459888841431929; 0.0 0.0 0.0; 1.1102230246251557e-16 -0.919777682863858 0.0; 0.22994442071596455 -0.919777682863858 -0.22994442071596444; 5.551115123125777e-17 -0.9197776828638579 -0.459888841431929])

In_bz = chull([-0.2901481927944896 0.7321728789887165 0.0; -0.511160535891603 0.511160535891603 -0.33611424804101847; -0.511160535891603 0.511160535891603 0.3361142480410185; -0.7321728789887165 0.29014819279448967 0.0; 0.511160535891603 0.511160535891603 -0.33611424804101847; 0.511160535891603 0.511160535891603 0.3361142480410185; 0.29014819279448967 0.7321728789887165 0.0; 0.7321728789887165 0.29014819279448967 0.0; -0.2901481927944896 -0.2901481927944896 -0.6722284960820372; -0.2901481927944896 0.2901481927944896 -0.6722284960820372; 0.2901481927944896 -0.2901481927944896 -0.6722284960820372; 0.2901481927944896 0.2901481927944896 -0.6722284960820372; 0.29014819279448956 0.29014819279448956 0.6722284960820373; 0.29014819279448956 -0.29014819279448956 0.6722284960820373; -0.29014819279448956 0.29014819279448956 0.6722284960820373; -0.29014819279448956 -0.29014819279448956 0.6722284960820373; 0.29014819279448967 -0.7321728789887165 0.0; 0.511160535891603 -0.511160535891603 0.3361142480410185; 0.511160535891603 -0.511160535891603 -0.33611424804101847; 0.7321728789887165 -0.29014819279448967 0.0; -0.511160535891603 -0.511160535891603 0.3361142480410185; -0.511160535891603 -0.511160535891603 -0.33611424804101847; -0.2901481927944896 -0.7321728789887165 0.0; -0.7321728789887165 -0.29014819279448967 0.0])
In_ibz = chull([0.0 0.0 0.0; 0.0 0.7321728789887165 0.0; 0.0 0.0 -0.6722284960820374; 0.0 0.2901481927944894 -0.6722284960820373; -0.2901481927944894 0.2901481927944894 -0.6722284960820373; -0.511160535891603 0.511160535891603 0.0; -0.29014819279448967 0.7321728789887165 0.0; -0.5111605358916032 0.5111605358916032 -0.33611424804101847])

K_bz = chull([5.5511151231257796e-17 -5.5511151231257796e-17 -0.6240438304791761; -0.31202191523958817 0.31202191523958805 -0.3120219152395881; 0.0 0.6240438304791761 0.0; 0.31202191523958805 -0.31202191523958805 0.31202191523958805; 0.6240438304791761 0.0 0.0; 0.31202191523958805 0.31202191523958805 0.31202191523958805; 0.31202191523958805 0.31202191523958805 -0.31202191523958805; 0.3120219152395882 -0.3120219152395881 -0.3120219152395881; -0.31202191523958817 0.31202191523958805 0.3120219152395881; 0.0 0.0 0.6240438304791761; -0.31202191523958817 -0.3120219152395881 0.31202191523958805; -0.31202191523958817 -0.3120219152395881 -0.31202191523958805; 5.5511151231257796e-17 -0.6240438304791761 -5.5511151231257796e-17; -0.6240438304791761 0.0 5.551115123125784e-17])
K_ibz = chull([0.0 0.0 0.0; -4.323549322550443e-17 0.31202191523958805 -0.31202191523958805; -0.31202191523958805 0.31202191523958817 -0.31202191523958817; -8.64709864510089e-17 -8.647098645100894e-17 -0.6240438304791763])

Li_bz = chull([-1.1102230246251565e-16 -5.551115123125783e-17 -0.9472757477392372; -0.4736378738696186 0.47363787386961853 -0.4736378738696186; 0.0 0.9472757477392372 0.0; 0.4736378738696186 -0.4736378738696186 0.4736378738696186; 0.9472757477392372 0.0 0.0; 0.4736378738696186 0.4736378738696186 0.4736378738696186; 0.4736378738696186 0.4736378738696186 -0.4736378738696186; 0.4736378738696186 -0.47363787386961864 -0.47363787386961864; -0.4736378738696186 0.47363787386961864 0.47363787386961864; -5.551115123125783e-17 -5.551115123125783e-17 0.9472757477392373; -0.4736378738696186 -0.47363787386961864 0.47363787386961875; -0.4736378738696186 -0.47363787386961864 -0.47363787386961864; -5.551115123125783e-17 -0.9472757477392372 0.0; -0.9472757477392372 0.0 5.551115123125783e-17])
Li_ibz = chull([0.0 0.0 0.0; 0.0 0.4736378738696186 -0.4736378738696186; -0.47363787386961864 0.47363787386961853 -0.47363787386961853; 0.0 0.0 -0.9472757477392372])

Na_bz = chull([0.0 0.0 -0.7749269628124452; -0.3874634814062226 0.3874634814062226 -0.3874634814062226; 0.0 0.7749269628124452 0.0; 0.3874634814062226 -0.3874634814062226 0.3874634814062226; 0.7749269628124452 0.0 0.0; 0.3874634814062226 0.3874634814062226 0.3874634814062226; 0.3874634814062226 0.3874634814062226 -0.3874634814062226; 0.3874634814062226 -0.3874634814062226 -0.3874634814062226; -0.3874634814062226 0.3874634814062226 0.3874634814062226; 0.0 0.0 0.7749269628124452; -0.3874634814062226 -0.3874634814062226 0.3874634814062226; -0.3874634814062226 -0.3874634814062226 -0.3874634814062226; 0.0 -0.7749269628124452 0.0; -0.7749269628124452 0.0 0.0])
Na_ibz = chull([0.0 0.0 0.0; 0.0 0.3874634814062226 -0.3874634814062226; -0.3874634814062226 0.3874634814062226 -0.3874634814062226; 0.0 0.0 -0.7749269628124452])

Pb_bz = chull([0.0 -0.6715890106758005 -0.33579450533790006; 0.0 -0.6715890106758005 0.33579450533790006; 0.33579450533790006 -0.6715890106758005 0.0; -0.3357945053379001 -0.6715890106758003 0.0; 0.0 0.33579450533790006 0.6715890106758005; 0.33579450533790006 0.0 0.6715890106758005; -0.3357945053379001 0.0 0.6715890106758003; 0.0 -0.3357945053379001 0.6715890106758003; 0.6715890106758003 -0.3357945053379001 0.0; 0.6715890106758003 0.0 -0.3357945053379001; 0.6715890106758003 0.0 0.3357945053379001; 0.6715890106758003 0.3357945053379001 0.0; 0.0 0.6715890106758003 0.3357945053379001; 0.0 0.6715890106758003 -0.3357945053379001; -0.3357945053379001 0.6715890106758003 0.0; 0.33579450533790006 0.6715890106758005 0.0; 0.0 -0.3357945053379001 -0.6715890106758003; -0.3357945053379001 0.0 -0.6715890106758003; 0.33579450533790006 0.0 -0.6715890106758005; 0.0 0.33579450533790006 -0.6715890106758005; -0.6715890106758005 0.33579450533790006 0.0; -0.6715890106758005 0.0 0.33579450533790006; -0.6715890106758005 0.0 -0.33579450533790006; -0.6715890106758005 -0.33579450533790006 0.0])
Pb_ibz = chull([0.0 -0.5036917580068503 -0.5036917580068504; 0.33579450533790023 -0.33579450533790023 -0.3357945053379003; 0.0 0.0 0.0; -1.3776071959892718e-16 -0.6715890106758005 0.0; 0.16789725266894998 -0.6715890106758003 -0.16789725266895011; -6.888035979946361e-17 -0.6715890106758005 -0.3357945053379002])

Rb_bz = chull([0.0 -5.551115123125785e-17 -0.5953312274073189; -0.2976656137036595 0.2976656137036594 -0.2976656137036595; 5.551115123125783e-17 0.595331227407319 5.551115123125783e-17; 0.29766561370365957 -0.29766561370365946 0.29766561370365946; 0.595331227407319 0.0 0.0; 0.2976656137036595 0.2976656137036595 0.2976656137036595; 0.29766561370365957 0.29766561370365946 -0.29766561370365946; 0.29766561370365946 -0.2976656137036595 -0.2976656137036594; -0.29766561370365935 0.2976656137036595 0.2976656137036595; 5.551115123125783e-17 5.551115123125783e-17 0.595331227407319; -0.2976656137036594 -0.2976656137036595 0.29766561370365957; -0.29766561370365935 -0.2976656137036595 -0.2976656137036594; 0.0 -0.5953312274073189 5.551115123125785e-17; -0.595331227407319 -2.77555756156289e-17 2.77555756156289e-17])
Rb_ibz = chull([0.0 0.0 0.0; 3.934844342871113e-17 0.29766561370365946 -0.2976656137036595; -0.29766561370365957 0.29766561370365935 -0.29766561370365957; 7.869688685742225e-17 7.869688685742228e-17 -0.5953312274073189])

Sn_bz = chull([2.775557561562892e-17 -8.326672684688678e-17 0.6780115716658311; -0.2850680689251662 -0.28506806892516623 0.5224754533735458; -0.570136137850332 -1.211498575340248e-16 0.36693933508126014; -0.28506806892516606 0.28506806892516595 0.5224754533735457; -1.5594397624517673e-16 -0.570136137850332 0.36693933508126036; 0.28506806892516606 -0.285068068925166 0.5224754533735457; 6.329782674123009e-17 0.570136137850332 0.3669393350812603; 0.2850680689251662 0.28506806892516606 0.5224754533735458; 0.5701361378503322 0.0 0.3669393350812604; 0.5701361378503323 0.0 -0.3669393350812603; 0.2850680689251662 0.28506806892516606 -0.5224754533735458; 6.32978267412301e-17 0.570136137850332 -0.36693933508126036; 0.28506806892516606 -0.28506806892516606 -0.5224754533735457; -1.5594397624517673e-16 -0.5701361378503321 -0.3669393350812604; -0.285068068925166 0.28506806892516606 -0.5224754533735458; -0.285068068925166 -0.28506806892516623 -0.522475453373546; -0.5701361378503321 -1.2114985753402484e-16 -0.3669393350812604; 5.551115123125784e-17 0.0 -0.678011571665831])
Sn_ibz = chull([-0.5701361378503322 0.0 -2.3339508271854204e-17; -0.2850680689251661 -0.2850680689251661 -2.3339508271854204e-17; 0.0 0.0 0.0; -0.5701361378503322 0.0 0.36693933508126025; 0.0 0.0 0.6780115716658311; -0.2850680689251661 -0.2850680689251661 0.5224754533735457])

Zn_bz = chull([-0.41589290938922413 0.7203476495697763 0.33606750608035785; -0.41589290938922413 0.7203476495697763 -0.33606750608035785; 0.4158929093892243 0.7203476495697763 0.33606750608035785; 0.4158929093892243 0.7203476495697763 -0.33606750608035785; 0.8317858187784489 0.0 0.33606750608035785; 0.8317858187784489 0.0 -0.33606750608035785; 0.4158929093892243 -0.7203476495697763 -0.33606750608035785; 0.4158929093892243 -0.7203476495697763 0.33606750608035785; -0.4158929093892243 -0.7203476495697763 0.33606750608035785; -0.4158929093892243 -0.7203476495697763 -0.33606750608035785; -0.831785818778449 -4.8329130850202184e-17 -0.3360675060803579; -0.831785818778449 -4.8329130850202184e-17 0.3360675060803579])
Zn_ibz = chull([0.0 0.0 0.0; 0.0 0.0 0.33606750608035785; -0.41589290938922446 0.7203476495697763 0.33606750608035785; -0.41589290938922446 0.7203476495697763 0.0; 2.3043821794774855e-16 0.7203476495697764 -1.1898599713114633e-16; 2.304382179477485e-16 0.7203476495697763 0.3360675060803579])

# Band energy and Fermi level solutions and approximate error computed with about 
# 3 million k-points with the rectangular method
Ag_flans = 2.1922390847785764
Al_flans = 11.591037691304805
Au_flans = 0.25853699543371234
Cs_flans = 1.3360115371222165
Cu_flans = 0.0
In_flans = 8.579352636903511
K_flans = 2.025607254792486
Li_flans = 3.9113730301061116
Na_flans = 3.1193508267702335
Pb_flans = 9.017164353350289
Rb_flans = 1.8548568671440286
Sn_flans = 5.905697491784268
Zn_flans = 5.524488853935539

Ag_flstd = 0.00029024054548876027
Al_flstd = 0.0008395537007797541
Au_flstd = 0.00035649434441836867
Cs_flstd = 6.975117482541303e-5
Cu_flstd = 0.0
In_flstd = 0.0004458156420962059
K_flstd = 0.0002935445102367859
Li_flstd = 0.0004358031904537981
Na_flstd = 0.00044732694710161134
Pb_flstd = 0.0007380499203445475
Rb_flstd = 0.00035086563370848473
Sn_flstd = 0.0013689154092370389
Zn_flstd = 0.00041898720333712655

Ag_beans = 0.6644229355784247
Al_beans = 22.99549912356057
Au_beans = -1.1340092325533375
Cs_beans = 0.11629735361316625
Cu_beans = 0.0
In_beans = 10.828585098946128
K_beans = 0.29421043930517937
Li_beans = 1.7553123751053519
Na_beans = 0.8668889251058358
Pb_beans = 12.009129106121884
Rb_beans = 0.23479467880160457
Sn_beans = 4.625508607056789
Zn_beans = 3.8962879178996266

Ag_bestd = 8.775063251184409e-6
Al_bestd = 3.0020754459305066e-5
Au_bestd = 2.9387131731311005e-6
Cs_bestd = 2.851175160523137e-5
Cu_bestd = 0.0
In_bestd = 0.00042706681206913376
K_bestd = 3.74162920869127e-6
Li_bestd = 0.0006671643075317917
Na_bestd = 9.023522804366152e-6
Pb_bestd = 2.9950783348248257e-5
Rb_bestd = 6.910086337481342e-7
Sn_bestd = 0.0018233084032896424
Zn_bestd = 4.026081648014471e-6

Ag_fermiarea = Ag_electrons/2*Ag_ibz.volume
Al_fermiarea = Al_electrons/2*Al_ibz.volume
Au_fermiarea = Au_electrons/2*Au_ibz.volume
Cs_fermiarea = Cs_electrons/2*Cs_ibz.volume
Cu_fermiarea = Cu_electrons/2*Cu_ibz.volume
In_fermiarea = In_electrons/2*In_ibz.volume
K_fermiarea = K_electrons/2*K_ibz.volume
Li_fermiarea = Li_electrons/2*Li_ibz.volume
Na_fermiarea = Na_electrons/2*Na_ibz.volume
Pb_fermiarea = Pb_electrons/2*Pb_ibz.volume
Rb_fermiarea = Rb_electrons/2*Rb_ibz.volume
Sn_fermiarea = Sn_electrons/2*Sn_ibz.volume
Zn_fermiarea = Zn_electrons/2*Zn_ibz.volume

atom_types = [0]
atom_pos = Array([0 0 0;]')
coordinates = "Cartesian"
convention = "ordinary"

v = Dict()
for m=epm_names
    v["ft"],v["pg"] = calc_spacegroup(eval(Symbol(m*"_latvecs")),atom_types,atom_pos,coordinates)
    @eval $(Symbol(m,"_pointgroup")) = v["pg"]
    @eval $(Symbol(m,"_frac_trans")) = v["ft"]
end

# 2D "toy" empirical pseudopotentials (the form factors are chosen at random)
m1rlat_type = "square"
m2rlat_type = "hexagonal"
m3rlat_type = "centered rectangular"
m4rlat_type = "rectangular"
m5rlat_type = "oblique"

# Brillouin zone and irreducible Brillouin zone for 2D models.
m1bz = chull([0.5 0.5; 0.5 -0.5; -0.5 -0.5; -0.5 0.5])
m1ibz = chull([0.0 0.0; 0.5 0.0; 0.5 0.5])

m2bz = chull([0.0 -0.5773502691896255; 0.5 0.2886751345948127; 0.5 -0.2886751345948127; -1.2819751242557097e-16 0.5773502691896257; -0.5000000000000001 0.2886751345948128; -0.5000000000000001 -0.28867513459481264])
m2ibz = chull([0.0 0.0; 0.2499999999999999 -0.4330127018922192; 0.0 -0.5773502691896256])

m3bz = chull([0.06611626088244185 -0.5867980452188124; 0.5 -0.31417082268360697; 0.5 0.31417082268360685; -0.5 -0.3141708226836071; -0.06611626088244209 0.5867980452188122; -0.5 0.31417082268360697])
m3ibz = chull([-0.5 -0.3141708226836068; 0.0 0.0; 0.28305813044122097 -0.4504844339512097; 0.0661162608824419 -0.5867980452188123])

m4bz = chull([0.5 1.0; -0.5 1.0; -0.5 -1.0; 0.5 -1.0])
m4ibz = chull([0.5 0.0; 0.0 0.0; 0.0 1.0; 0.5 1.0])

m5bz = chull([0.09999999999999995 0.6350852961085884; 0.5 0.4041451884327381; 0.5 -0.40414518843273806; -0.5 0.4041451884327381; -0.1000000000000001 -0.6350852961085884; -0.5 -0.4041451884327381])
m5ibz = chull([-0.5 0.4041451884327381; 0.5 -0.07872958216222165; -0.5 0.07872958216222165; 0.5 0.4041451884327381; 0.10000000000000002 0.6350852961085884])

# Model 1 - square symmetry
atom_types = [0]
atom_pos = Array([0 0;]')
coordinates = "Cartesian" 

convention = "ordinary"
m1real_latvecs = [1 0; 0 1]
(m1frac_trans,m1pointgroup) = calc_spacegroup(m1real_latvecs,atom_types,atom_pos,
    coordinates)
m1recip_latvecs = get_recip_latvecs(m1real_latvecs,convention)
m1dist_ff = [[1.00,2.00],[-0.23,0.12]]
m1rules = [1.00 => -0.23, 1.41 => 0.12]
m1electrons1 = 6

m1cutoff = 6.1
m1fermiarea1 = m1electrons1/2*m1ibz.volume
m1fermilevel1 = 0.9381309375519309
m1bandenergy1 = 1.0423513259405526

m1electrons2 = 7
m1fermiarea2 = m1electrons2/2*m1ibz.volume
m1fermilevel2 = 1.1057574662181007
m1bandenergy2 = 1.5535874027360785

m1electrons3 = 8
m1fermiarea3 = m1electrons3/2*m1ibz.volume
m1fermilevel3 = 1.258616280256323
m1bandenergy3 = 2.145219996962875

# Model 2 - hexagonal symmetry
convention = "ordinary"
m2recip_latvecs = [0.5 0.5; 0.8660254037844386 -0.8660254037844386]
m2real_latvecs = get_recip_latvecs(m2recip_latvecs,convention)
(m2frac_trans,m2pointgroup) = calc_spacegroup(m2real_latvecs,atom_types,atom_pos,
    coordinates)
m2dist_ff = [[1.00,3.00,4.00],[0.39,0.23,-0.11]]
m2rules = [1.0 => 0.39, 1.73 => 0.23, 2.0 => -0.11]
m2cutoff = 5.9

m2electrons1 = 5
m2fermiarea1 = m2electrons1/2*m2ibz.volume
m2fermilevel1 = 0.06138423898212197
m2bandenergy1 = -0.4302312741509512

m2electrons2 = 7
m2fermiarea2 = m2electrons2/2*m2ibz.volume
m2fermilevel2 = 0.9021827685803184
m2bandenergy2 = -0.024599327665460413

m2electrons3 = 8
m2fermiarea3 = m2electrons3/2*m2ibz.volume
m2fermilevel3 = 0.9968615721784458
m2bandenergy3 = 0.38884262264868563

# Model 3 - centered rectangular symmetry
convention = "ordinary"
m3recip_latvecs = [0.4338837391175581 1.0; 0.9009688679024191 0.0]
m3real_latvecs = get_recip_latvecs(m3recip_latvecs,convention)
(m3frac_trans,m3pointgroup) = calc_spacegroup(m3real_latvecs,atom_types,atom_pos,
    coordinates)
m3dist_ff = [[1.00,1.13,2.87],[-0.27,0.2,-0.33]]
m3rules = [1.0 => -0.27, 1.06 => 0.2, 1.69 => -0.33]
m3cutoff = 5.95

m3electrons1 = 5
m3fermiarea1 = m3electrons1/2*m3ibz.volume
m3fermilevel1 = 0.5833433206577795
m3bandenergy1 = 0.0035066586253235665

m3electrons2 = 7
m3fermiarea2 = m3electrons2/2*m3ibz.volume
m3fermilevel2 = 0.9911138305912597
m3bandenergy2 = 0.71444638735834

m3electrons3 = 8
m3fermiarea3 = m3electrons3/2*m3ibz.volume
m3fermilevel3 = 1.1117071929086504
m3bandenergy3 = 1.1860046687293682

# Model 4 - rectangular symmetry
convention = "ordinary"
m4recip_latvecs = [1 0; 0 2]
m4real_latvecs = get_recip_latvecs(m4recip_latvecs,convention)
(m4frac_trans,m4pointgroup) = calc_spacegroup(m4real_latvecs,atom_types,atom_pos,
    coordinates)
m4dist_ff = [[1.00,4.00,5.00],[0.39,-0.11,0.11]]
m4rules = [1.0 => 0.39, 2.0 => -0.11, 2.24 => 0.11]
m4cutoff = 8.6

m4electrons1 = 6
m4fermiarea1 = m4electrons1/2*m4ibz.volume
m4fermilevel1 = 1.9034249381001005
m4bandenergy1 = 5.1056578173306795

m4electrons2 = 7
m4fermiarea2 = m4electrons2/2*m4ibz.volume
m4fermilevel2 = 2.2266488438956333
m4bandenergy2 = 7.1685536634386136

m4electrons3 = 8
m4fermiarea3 = m4electrons3/2*m4ibz.volume
m4fermilevel3 = 2.551004975931985
m4bandenergy3 = 9.555193758896971

# Model 5 - oblique symmetry
convention = "ordinary"
m5recip_latvecs = [1.0 -0.4; 0.0 1.0392304845413265]
m5real_latvecs = get_recip_latvecs(m5recip_latvecs,convention)
(m5frac_trans,m5pointgroup) = calc_spacegroup(m5real_latvecs,atom_types,atom_pos,
    coordinates)
m5dist_ff = [[1.0,1.24,1.44],[0.42,0.02,-0.18]]
m5rules = [1.0 => 0.42, 1.11 => 0.02, 1.2 => -0.18]
m5cutoff = 6.3

m5electrons1 = 5
m5fermiarea1 = m5electrons1/2*m5ibz.volume
m5fermilevel1 = 0.7916464535133585
m5bandenergy1 = 0.47750270146629903

m5electrons2 = 7
m5fermiarea2 = m5electrons2/2*m5ibz.volume
m5fermilevel2 = 1.1470444743280181
m5bandenergy2 = 1.4588171623200643

m5electrons3 = 9
m5fermiarea3 = m5electrons3/2*m5ibz.volume
m5fermilevel3 = 1.4883816210907215
m5bandenergy3 = 2.8258962133639556

@doc """
    epm₋model2D(energy_conv,sheets,real_latvecs,recip_latvecs,bz,ibz,pointgroup,
        frac_trans,rules,cutoff,electrons,fermiarea,fermilevel,bandenergy)

A container for all the information about the 2D empirical pseudopotential model(s).
"""
mutable struct epm₋model2D
    energy_conv::Real
    sheets::Int 
    atom_types::Vector{<:Int}
    atom_pos::Matrix{<:Real}
    coordinates::String
    convention::String

    real_latvecs::Matrix{<:Real}
    recip_latvecs::Matrix{<:Real}
    bz::Chull{<:Real}
    ibz::Chull{<:Real}
    pointgroup::Vector{Matrix{Float64}}
    frac_trans::Vector{Vector{Float64}}
    dist_ff::Vector{Vector{Float64}}
    rules::Vector{Pair{Float64, Float64}}
    cutoff::Real
    rlat_type::String
    
    electrons::Real
    fermiarea::Real
    fermilevel::Real
    bandenergy::Real
end

energy_conv = 1
sheets = 10
atom_types = [0]
atom_pos = Array([0 0 0;]')
coordinates = "Cartesian" 
convention = "ordinary"
vars₀ = ["energy_conv","sheets","atom_types","atom_pos","coordinates",
        "convention"]
vars₁ = ["real_latvecs","recip_latvecs","bz","ibz","pointgroup","frac_trans",
        "dist_ff","rules","cutoff","rlat_type"]
vars₂ = ["electrons","fermiarea","fermilevel","bandenergy"];
v = Dict()
for i=1:5
    [v[var] = (var |> Symbol |> eval) for var=vars₀]
    [v[var] = ("m"*string(i)*var |> Symbol |> eval) for var=vars₁]
    for j=1:3
        [v[var] = ("m"*string(i)*var*string(j) |> Symbol |> eval) for var=vars₂]
        name = "m"*string(i)*string(j)        
        @eval $(Symbol(name)) = epm₋model2D([v[var] for var=[vars₀; vars₁; vars₂]]...)
    end
end

@doc """
    epm₋model()

A container for all the information about the empirical pseudopotential models.
"""
mutable struct epm₋model
    energy_conv::Real 
    sym_offset::Vector{<:Real}
    atom_types::Vector{<:Int}
    atom_pos::Matrix{<:Real}
    coordinates::String
    convention::String

    sheets::Int
    name::String
    lat_type::String
    lat_constants::Vector{<:Real}
    lat_angles::Vector{<:Real}
    real_latvecs::Matrix{<:Real}
    rlat_type::String
    recip_latvecs::Matrix{<:Real}
    pointgroup::Vector{Matrix{Float64}}
    frac_trans::Vector{Vector{Float64}}
    
    bz::Chull{<:Real}
    ibz::Chull{<:Real}    
    dist_ff::Vector{Vector{Float64}}
    rules::Vector{Pair{Float64, Float64}}
    electrons::Real
    cutoff::Real
    fermiarea::Real
    fermilevel::Real
    fl_error::Real
    bandenergy::Real
    be_error::Real
end

energy_conv = RytoeV
sheets = 10
atom_types = [0]
atom_pos = Array([0 0 0;]')
coordinates = "Cartesian" 
convention = "angular"
vars₀ = ["energy_conv","offset","atom_types","atom_pos","coordinates","convention"]
vars₁ = ["sheets","name","type","abc","αβγ","latvecs","rtype","rlatvecs","pointgroup","frac_trans",
        "bz","ibz","dist_ff","rules","electrons","cutoff","fermiarea","flans","flstd","beans","bestd"]
v = Dict()
offset = [0.0,0,0]
for m=epm_names
    offset = sym_offset[eval(Symbol(m,"_rtype"))]
    [v[var] = eval(Symbol(var)) for var=vars₀]
    [v[var] = eval(Symbol(m,"_",var)) for var=vars₁]
    @eval $(Symbol(m,"_epm")) = epm₋model([v[var] for var=[vars₀;vars₁]]...)
end
        

@doc """
    eval_epm(kpoint,rbasis,rules,cutoff,sheets,energy_conversion_factor;rtol,
        atol,func)

Evaluate an empirical pseudopotential at a k-point.

# Arguments
- `kpoint::AbstractVector{<:Real}:` a point at which the EPM is evaluated.
- `rbasis::AbstractMatrix{<:Real}`: the reciprocal lattice basis as columns of
    a 3x3 real array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets::Int`: the number of eigenenergies returned.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(rbasis))))`: a relative tolerance for
    finite precision comparisons. This is used for identifying points within a
    circle or sphere in the Fourier expansion.
- `atol::Real=1e-9`: an absolute tolerance for finite precision comparisons.
- `func::Union{Nothing,Function}=nothing`: a k-point independent EPM.

# Returns
- `::AbstractVector{<:Real}`: a list of eigenenergies

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm
kpoint = [0,0,0]
rlatvecs = [1 0 0; 0 1 0; 0 0 1]
rules = Dict(1.00 => .01, 2.00 => 0.015)
cutoff = 3.0
sheets = 1:10
eval_epm(kpoint, rlatvecs, rules, cutoff, sheets)
# output
10-element Array{Float64,1}:
 -0.012572222255690903
 13.392395133818168
 13.392395133818248
 13.392395133818322
 13.803213112862565
 13.803213112862627
 13.803213665491697
 26.79812229071137
 26.7981222907114
 26.798122290711415
```
"""
function eval_epm(kpoint::AbstractVector{<:Real},
    rbasis::AbstractMatrix{<:Real}, rules::Dict{Float64,Float64}, cutoff::Real,
    sheets::Int,energy_conversion_factor::Real=RytoeV;
    rtol::Real=sqrt(eps(float(maximum(rbasis)))),
    atol::Real=1e-9,
    func::Union{Nothing,Function}=nothing)::AbstractVector{<:Real}

    if !(func == nothing)
        return eval_epm(func,kpoint...,sheets)
    end

    if length(kpoint) == 3
        rlatpts = sample_sphere(rbasis,cutoff,kpoint,rtol=rtol,atol=atol)
    elseif length(kpoint) == 2
        rlatpts = sample_circle(rbasis,cutoff,kpoint,rtol=rtol,atol=atol)
    else
        throw(ArgumentError("The k-point may only have 2 or 3 elements."))
    end

    npts = size(rlatpts,2)
    if npts < sheets[end]
        error("The cutoff is too small for the requested number of sheets. The"*
            " number of terms in the expansion is $npts.")
    end
    ham=zeros(Float64,npts,npts)
    dist = 0.0
    for i=1:npts, j=i:npts
        if i==j
            # ham[i,j] = norm(kpoint + rlatpts[:,i])^2
            ham[i,j] = dot(kpoint + rlatpts[:,i],kpoint + rlatpts[:,i])
        else
            dist = round(norm(rlatpts[:,i] - rlatpts[:,j]),digits=2)
            if haskey(rules,dist)
                ham[i,j] = rules[dist]
            end
        end
    end

    eigvals(Symmetric(ham))[1:sheets]*energy_conversion_factor
end


@doc """
    eval_epm(kpoints,rbasis,rules,cutoff,sheets,energy_conversion_factor;rtol,atol,func)

Evaluate an empirical pseudopotential at each point in an array.

# Arguments
- `kpoints::AbstractMatrix{<:Real}`: an array of k-points as columns of an
    array.
"""
function eval_epm(kpoints::AbstractMatrix{<:Real},
    rbasis::AbstractMatrix{<:Real}, rules::Dict{Float64,Float64}, cutoff::Real,
    sheets::Int,energy_conversion_factor::Real=RytoeV;
    rtol::Real=sqrt(eps(float(maximum(rbasis)))),
    atol::Real=1e-9,
    func::Union{Nothing,Function}=nothing)::AbstractMatrix{<:Real}

    if !(func == nothing)
        eval_epm(func,kpoints,sheets)
    end

    mapslices(x->eval_epm(x,rbasis,rules,cutoff,sheets,energy_conversion_factor;
        rtol=rtol,atol=atol),kpoints,dims=1)
end

@doc """
    eval_epm(kpoint,epm;rtol,atol,func,sheets)

Evaluate an empirical pseudopotential at a k-point.

# Arguments
- `kpoint::AbstractVector{<:Real}`: a k-point at which the EPM is evaluated.
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs))))`: a relative tolerance.
- `atol::Real=1e-9`: an absolute tolerances
- `sheets::Integer=-1`: the number of sheets for the k-point independent EPM.

# Returns
- `::AbstractVector{<:Real}`: the eigenenergies at the k-point for the EPM.

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm,Al_epm
eval_epm([0,0,0],Al_epm)
# output 
10-element Vector{Float64}:
 -0.05904720925479707
 26.68476356943067
 26.684763569434374
 26.68476356943802
 26.713073023266258
 28.17466342570331
 28.24043839738628
 28.24043839738743
 28.240438397392488
 36.545883436839766
```
"""
function eval_epm(kpoint::AbstractVector{<:Real},
    epm::Union{epm₋model2D,epm₋model};
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))),
    atol::Real=1e-9,sheets::Integer=-1)::AbstractVector{<:Real}

    if length(kpoint) == 3
        rlatpts = sample_sphere(epm.recip_latvecs,epm.cutoff,kpoint;rtol=rtol,
            atol=atol)
    elseif length(kpoint) == 2
        rlatpts = sample_circle(epm.recip_latvecs,epm.cutoff,kpoint;rtol=rtol,
            atol=atol)
    else
        throw(ArgumentError("The k-point may only have 2 or 3 elements."))
    end

    npts = size(rlatpts,2)
    if npts < sheets
        error("The cutoff is too small for the requested number of sheets. The"*
            " number of terms in the expansion is $npts.")
    end

    if sheets < 0
        sheets = epm.sheets
    end

    ham=zeros(Float64,npts,npts)
    rules = epm.rules
    maxff = maximum([x[2] for x=rules])
    maxd = maximum([x[1] for x=rules]) + 1e-3
    pairwise!(ham, SqEuclidean(), rlatpts, dims=2)
    replace!(x -> x > maxd ? 0 : x, ham)
    ind = findall(!iszero, ham)
    ham[ind] = replace(round.(ham[ind],digits=2),rules...)
    ham[ind] = replace(x -> x > maxff ? 0 : x, ham[ind])

    if length(kpoint) == 2
        for i=1:npts
            ham[i,i] = (kpoint[1] + rlatpts[1,i])^2 + (kpoint[2] + rlatpts[2,i])^2
        end
    else
        for i=1:npts
            ham[i,i] = (kpoint[1] + rlatpts[1,i])^2 + (kpoint[2] + rlatpts[2,i])^2 + (kpoint[3] + rlatpts[3,i])^2
        end
    end
    eigs(SparseMatrixCSC(ham),ritzvec=false,nev=2*sheets,which=:SR)[1][1:sheets]*epm.energy_conv 
end

@doc """
    eval_epm(kpoints,epm,rtol,atol,func,sheets)

Evaluate an EPM an many k-points.

# Arguments
- `kpoints::AbstractMatrix{<:Real}`: a matrix whose columns are k-point points.
- `epm::Union{epm₋model2D,epm₋model}`: an EPM model.
- `rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs))))`: a relative tolerance.
- `atol::Real=1e-9`: an absolute tolerance.
- `func::Union{Nothing,Function}=nothing`: a k-point independent EPM.
- `sheets::Integer=10`: the number of sheets for the k-point independent EPM.

# Returns
- `::AbstractMatrix{<:Real}`: a matrix whose columns are eigenvalues at the 
    k-point in the same column.

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm, Al_epm
eval_epm([0 0; 0 1; 0 0],Al_epm)
# output
10×2 Matrix{Float64}:
 -0.0590472   5.54208
 26.6848     13.5071
 26.6848     18.7083
 26.6848     18.7083
 26.7131     18.7174
 28.1747     18.7612
 28.2404     41.9315
 28.2404     41.9643
 28.2404     42.2674
 36.5459     42.2674
```
"""
function eval_epm(kpoints::AbstractMatrix{<:Real},
    epm::Union{epm₋model2D,epm₋model};
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))),
    atol=1e-9,sheets::Integer=-1)::AbstractMatrix{<:Real}

    mapslices(x->eval_epm(x,epm,rtol=rtol,atol=atol,sheets=sheets),
    kpoints,dims=1)
end

@doc """
    eval_epm(kpoint,func)

Evaluate a *k*-point independent EPM `func` at a provided *k*-point.
"""
function eval_epm(func::Function,kpoint::AbstractVector{<:Real},s::Integer)
    func(kpoint...,s)
end

@doc """
    eval_epm(kpoints,func)

Evaluate a *k*-point independent EPM `func` at *k*-points that are columns of a matrix.
"""
function eval_epm(func::Function,kpoints::AbstractMatrix{<:Real},s::Integer)
    mapslices(x->eval_epm(func,x,s),kpoints,dims=1)
end

#=
k-point independent models have at least 15 terms in the Fourier expansion and 
no more 23 (Zn). Because the number of terms is so small, some pseudopotential 
form factors of a model may not appear in the Hamiltonian of the model.
=#

@doc """
    Ag(x,y,z,s)

A k-point independent EPM for Ag.
"""
Ag(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-0.813873564743926 + x)*(-0.813873564743926 + x) + (-0.813873564743926 + y)*(-0.813873564743926 + y) + (-0.813873564743926 + z)*(-0.813873564743926 + z) 0.195000000000000 0.195000000000000 0 0.195000000000000 0 0 0.195000000000000 0 0 0 0 0 0 0.121000000000000; 0.195000000000000 (-1.62774712948785 + x)*(-1.62774712948785 + x) + y*y + z*z 0 0.195000000000000 0 0.195000000000000 0 0 0.195000000000000 0 0 0 0 0 0; 0.195000000000000 0 (-1.62774712948785 + y)*(-1.62774712948785 + y) + x*x + z*z 0.195000000000000 0 0 0.195000000000000 0 0 0.195000000000000 0 0 0 0 0; 0 0.195000000000000 0.195000000000000 (-0.813873564743926 + x)*(-0.813873564743926 + x) + (-0.813873564743926 + y)*(-0.813873564743926 + y) + (0.813873564743926 + z)*(0.813873564743926 + z) 0 0 0 0.195000000000000 0 0 0.195000000000000 0.121000000000000 0 0 0; 0.195000000000000 0 0 0 (-1.62774712948785 + z)*(-1.62774712948785 + z) + x*x + y*y 0.195000000000000 0.195000000000000 0 0 0 0 0.195000000000000 0 0 0; 0 0.195000000000000 0 0 0.195000000000000 (-0.813873564743926 + x)*(-0.813873564743926 + x) + (-0.813873564743926 + z)*(-0.813873564743926 + z) + (0.813873564743926 + y)*(0.813873564743926 + y) 0 0.195000000000000 0 0.121000000000000 0 0 0.195000000000000 0 0; 0 0 0.195000000000000 0 0.195000000000000 0 (-0.813873564743926 + y)*(-0.813873564743926 + y) + (-0.813873564743926 + z)*(-0.813873564743926 + z) + (0.813873564743926 + x)*(0.813873564743926 + x) 0.195000000000000 0.121000000000000 0 0 0 0 0.195000000000000 0; 0.195000000000000 0 0 0.195000000000000 0 0.195000000000000 0.195000000000000 x*x + y*y + z*z 0.195000000000000 0.195000000000000 0 0.195000000000000 0 0 0.195000000000000; 0 0.195000000000000 0 0 0 0 0.121000000000000 0.195000000000000 (-0.813873564743926 + x)*(-0.813873564743926 + x) + (0.813873564743926 + y)*(0.813873564743926 + y) + (0.813873564743926 + z)*(0.813873564743926 + z) 0 0.195000000000000 0 0.195000000000000 0 0; 0 0 0.195000000000000 0 0 0.121000000000000 0 0.195000000000000 0 (-0.813873564743926 + y)*(-0.813873564743926 + y) + (0.813873564743926 + x)*(0.813873564743926 + x) + (0.813873564743926 + z)*(0.813873564743926 + z) 0.195000000000000 0 0 0.195000000000000 0; 0 0 0 0.195000000000000 0 0 0 0 0.195000000000000 0.195000000000000 (1.62774712948785 + z)*(1.62774712948785 + z) + x*x + y*y 0 0 0 0.195000000000000; 0 0 0 0.121000000000000 0.195000000000000 0 0 0.195000000000000 0 0 0 (-0.813873564743926 + z)*(-0.813873564743926 + z) + (0.813873564743926 + x)*(0.813873564743926 + x) + (0.813873564743926 + y)*(0.813873564743926 + y) 0.195000000000000 0.195000000000000 0; 0 0 0 0 0 0.195000000000000 0 0 0.195000000000000 0 0 0.195000000000000 (1.62774712948785 + y)*(1.62774712948785 + y) + x*x + z*z 0 0.195000000000000; 0 0 0 0 0 0 0.195000000000000 0 0 0.195000000000000 0 0.195000000000000 0 (1.62774712948785 + x)*(1.62774712948785 + x) + y*y + z*z 0.195000000000000; 0.121000000000000 0 0 0 0 0 0 0.195000000000000 0 0 0.195000000000000 0 0.195000000000000 0.195000000000000 (0.813873564743926 + x)*(0.813873564743926 + x) + (0.813873564743926 + y)*(0.813873564743926 + y) + (0.813873564743926 + z)*(0.813873564743926 + z)])[1:s]

@doc """
    Al(x,y,z,s)

A k-point independent EPM for Al.
"""
Al(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-0.82107382091626 + x)*(-0.82107382091626 + x) + (-0.82107382091626 + y)*(-0.82107382091626 + y) + (-0.82107382091626 + z)*(-0.82107382091626 + z) 0.0179000000000000 0.0179000000000000 0 0.0179000000000000 0 0 0.0179000000000000 0 0 0 0 0 0 0.0562000000000000; 0.0179000000000000 (-1.64214764183252 + x)*(-1.64214764183252 + x) + (-2.22044604925031e-16 + y)*(-2.22044604925031e-16 + y) + (1.11022302462516e-16 + z)*(1.11022302462516e-16 + z) 0 0.0179000000000000 0 0.0179000000000000 0 0 0.0179000000000000 0 0 0 0 0 0; 0.0179000000000000 0 (-1.64214764183252 + y)*(-1.64214764183252 + y) + (-2.22044604925031e-16 + x)*(-2.22044604925031e-16 + x) + (1.11022302462516e-16 + z)*(1.11022302462516e-16 + z) 0.0179000000000000 0 0 0.0179000000000000 0 0 0.0179000000000000 0 0 0 0 0; 0 0.0179000000000000 0.0179000000000000 (-0.82107382091626 + x)*(-0.82107382091626 + x) + (-0.82107382091626 + y)*(-0.82107382091626 + y) + (0.82107382091626 + z)*(0.82107382091626 + z) 0 0 0 0.0179000000000000 0 0 0.0179000000000000 0.0562000000000000 0 0 0; 0.0179000000000000 0 0 0 (-1.64214764183252 + z)*(-1.64214764183252 + z) + (-3.33066907387547e-16 + x)*(-3.33066907387547e-16 + x) + (-3.33066907387547e-16 + y)*(-3.33066907387547e-16 + y) 0.0179000000000000 0.0179000000000000 0 0 0 0 0.0179000000000000 0 0 0; 0 0.0179000000000000 0 0 0.0179000000000000 (-0.82107382091626 + x)*(-0.82107382091626 + x) + (-0.82107382091626 + z)*(-0.82107382091626 + z) + (0.821073820916259 + y)*(0.821073820916259 + y) 0 0.0179000000000000 0 0.0562000000000000 0 0 0.0179000000000000 0 0; 0 0 0.0179000000000000 0 0.0179000000000000 0 (-0.82107382091626 + y)*(-0.82107382091626 + y) + (-0.82107382091626 + z)*(-0.82107382091626 + z) + (0.821073820916259 + x)*(0.821073820916259 + x) 0.0179000000000000 0.0562000000000000 0 0 0 0 0.0179000000000000 0; 0.0179000000000000 0 0 0.0179000000000000 0 0.0179000000000000 0.0179000000000000 x*x + y*y + z*z 0.0179000000000000 0.0179000000000000 0 0.0179000000000000 0 0 0.0179000000000000; 0 0.0179000000000000 0 0 0 0 0.0562000000000000 0.0179000000000000 (-0.821073820916259 + x)*(-0.821073820916259 + x) + (0.82107382091626 + z)*(0.82107382091626 + z) + (0.82107382091626 + y)*(0.82107382091626 + y) 0 0.0179000000000000 0 0.0179000000000000 0 0; 0 0 0.0179000000000000 0 0 0.0562000000000000 0 0.0179000000000000 0 (-0.821073820916259 + y)*(-0.821073820916259 + y) + (0.82107382091626 + z)*(0.82107382091626 + z) + (0.82107382091626 + x)*(0.82107382091626 + x) 0.0179000000000000 0 0 0.0179000000000000 0; 0 0 0 0.0179000000000000 0 0 0 0 0.0179000000000000 0.0179000000000000 (3.33066907387547e-16 + x)*(3.33066907387547e-16 + x) + (3.33066907387547e-16 + y)*(3.33066907387547e-16 + y) + (1.64214764183252 + z)*(1.64214764183252 + z) 0 0 0 0.0179000000000000; 0 0 0 0.0562000000000000 0.0179000000000000 0 0 0.0179000000000000 0 0 0 (-0.82107382091626 + z)*(-0.82107382091626 + z) + (0.82107382091626 + x)*(0.82107382091626 + x) + (0.82107382091626 + y)*(0.82107382091626 + y) 0.0179000000000000 0.0179000000000000 0; 0 0 0 0 0 0.0179000000000000 0 0 0.0179000000000000 0 0 0.0179000000000000 (-1.11022302462516e-16 + z)*(-1.11022302462516e-16 + z) + (2.22044604925031e-16 + x)*(2.22044604925031e-16 + x) + (1.64214764183252 + y)*(1.64214764183252 + y) 0 0.0179000000000000; 0 0 0 0 0 0 0.0179000000000000 0 0 0.0179000000000000 0 0.0179000000000000 0 (-1.11022302462516e-16 + z)*(-1.11022302462516e-16 + z) + (2.22044604925031e-16 + y)*(2.22044604925031e-16 + y) + (1.64214764183252 + x)*(1.64214764183252 + x) 0.0179000000000000; 0.0562000000000000 0 0 0 0 0 0 0.0179000000000000 0 0 0.0179000000000000 0 0.0179000000000000 0.0179000000000000 (0.82107382091626 + z)*(0.82107382091626 + z) + (0.82107382091626 + x)*(0.82107382091626 + x) + (0.82107382091626 + y)*(0.82107382091626 + y)])[1:s]

@doc """
    Au(x,y,z,s)

A k-point independent EPM for Au.
"""
Au(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-0.815288684804078 + x)*(-0.815288684804078 + x) + (-0.815288684804078 + y)*(-0.815288684804078 + y) + (-0.815288684804078 + z)*(-0.815288684804078 + z) 0.252000000000000 0.252000000000000 0 0.252000000000000 0 0 0.252000000000000 0 0 0 0 0 0 0.152000000000000; 0.252000000000000 (-1.63057736960816 + x)*(-1.63057736960816 + x) + y*y + z*z 0 0.252000000000000 0 0.252000000000000 0 0 0.252000000000000 0 0 0 0 0 0; 0.252000000000000 0 (-1.63057736960816 + y)*(-1.63057736960816 + y) + x*x + z*z 0.252000000000000 0 0 0.252000000000000 0 0 0.252000000000000 0 0 0 0 0; 0 0.252000000000000 0.252000000000000 (-0.815288684804078 + x)*(-0.815288684804078 + x) + (-0.815288684804078 + y)*(-0.815288684804078 + y) + (0.815288684804078 + z)*(0.815288684804078 + z) 0 0 0 0.252000000000000 0 0 0.252000000000000 0.152000000000000 0 0 0; 0.252000000000000 0 0 0 (-1.63057736960816 + z)*(-1.63057736960816 + z) + x*x + y*y 0.252000000000000 0.252000000000000 0 0 0 0 0.252000000000000 0 0 0; 0 0.252000000000000 0 0 0.252000000000000 (-0.815288684804078 + x)*(-0.815288684804078 + x) + (-0.815288684804078 + z)*(-0.815288684804078 + z) + (0.815288684804078 + y)*(0.815288684804078 + y) 0 0.252000000000000 0 0.152000000000000 0 0 0.252000000000000 0 0; 0 0 0.252000000000000 0 0.252000000000000 0 (-0.815288684804078 + y)*(-0.815288684804078 + y) + (-0.815288684804078 + z)*(-0.815288684804078 + z) + (0.815288684804078 + x)*(0.815288684804078 + x) 0.252000000000000 0.152000000000000 0 0 0 0 0.252000000000000 0; 0.252000000000000 0 0 0.252000000000000 0 0.252000000000000 0.252000000000000 x*x + y*y + z*z 0.252000000000000 0.252000000000000 0 0.252000000000000 0 0 0.252000000000000; 0 0.252000000000000 0 0 0 0 0.152000000000000 0.252000000000000 (-0.815288684804078 + x)*(-0.815288684804078 + x) + (0.815288684804078 + y)*(0.815288684804078 + y) + (0.815288684804078 + z)*(0.815288684804078 + z) 0 0.252000000000000 0 0.252000000000000 0 0; 0 0 0.252000000000000 0 0 0.152000000000000 0 0.252000000000000 0 (-0.815288684804078 + y)*(-0.815288684804078 + y) + (0.815288684804078 + x)*(0.815288684804078 + x) + (0.815288684804078 + z)*(0.815288684804078 + z) 0.252000000000000 0 0 0.252000000000000 0; 0 0 0 0.252000000000000 0 0 0 0 0.252000000000000 0.252000000000000 (1.63057736960816 + z)*(1.63057736960816 + z) + x*x + y*y 0 0 0 0.252000000000000; 0 0 0 0.152000000000000 0.252000000000000 0 0 0.252000000000000 0 0 0 (-0.815288684804078 + z)*(-0.815288684804078 + z) + (0.815288684804078 + x)*(0.815288684804078 + x) + (0.815288684804078 + y)*(0.815288684804078 + y) 0.252000000000000 0.252000000000000 0; 0 0 0 0 0 0.252000000000000 0 0 0.252000000000000 0 0 0.252000000000000 (1.63057736960816 + y)*(1.63057736960816 + y) + x*x + z*z 0 0.252000000000000; 0 0 0 0 0 0 0.252000000000000 0 0 0.252000000000000 0 0.252000000000000 0 (1.63057736960816 + x)*(1.63057736960816 + x) + y*y + z*z 0.252000000000000; 0.152000000000000 0 0 0 0 0 0 0.252000000000000 0 0 0.252000000000000 0 0.252000000000000 0.252000000000000 (0.815288684804078 + x)*(0.815288684804078 + x) + (0.815288684804078 + y)*(0.815288684804078 + y) + (0.815288684804078 + z)*(0.815288684804078 + z)])[1:s]

@doc """
    Cs(x,y,z,s)

A k-point independent EPM for Cs.
"""
Cs(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.08285973169371 + x)*(-1.08285973169371 + x) + y*y + z*z 0 0 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 0 0 0; 0 (-0.541429865846855 + x)*(-0.541429865846855 + x) + (-0.541429865846855 + y)*(-0.541429865846855 + y) + z*z 0 0 0 -0.0300000000000000 0 0 0 0 -0.0300000000000000 0 -0.0300000000000000 -0.0300000000000000 -0.0300000000000000 0 -0.0300000000000000 0 0; 0 0 (-0.541429865846855 + x)*(-0.541429865846855 + x) + (0.541429865846855 + z)*(0.541429865846855 + z) + y*y -0.0300000000000000 0 0 0 0 -0.0300000000000000 0 0 -0.0300000000000000 0 0 -0.0300000000000000 -0.0300000000000000 0 -0.0300000000000000 0; 0 0 -0.0300000000000000 (-1.08285973169371 + y)*(-1.08285973169371 + y) + x*x + z*z 0 0 -0.0300000000000000 0 0 0 0 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 0; -0.0300000000000000 0 0 0 (-0.541429865846855 + y)*(-0.541429865846855 + y) + (0.541429865846855 + z)*(0.541429865846855 + z) + x*x 0 -0.0300000000000000 -0.0300000000000000 0 0 0 0 0 0 0 0 -0.0300000000000000 -0.0300000000000000 -0.0300000000000000; 0 -0.0300000000000000 0 0 0 (1.08285973169371 + z)*(1.08285973169371 + z) + x*x + y*y 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 0 0 0 0 -0.0300000000000000 0; 0 0 0 -0.0300000000000000 -0.0300000000000000 0 (-0.541429865846855 + x)*(-0.541429865846855 + x) + (-0.541429865846855 + z)*(-0.541429865846855 + z) + y*y 0 0 0 -0.0300000000000000 -0.0300000000000000 0 0 0 -0.0300000000000000 0 -0.0300000000000000 0; 0 0 0 0 -0.0300000000000000 -0.0300000000000000 0 (-0.541429865846855 + x)*(-0.541429865846855 + x) + (0.541429865846855 + y)*(0.541429865846855 + y) + z*z -0.0300000000000000 0 0 0 -0.0300000000000000 -0.0300000000000000 0 0 -0.0300000000000000 0 0; -0.0300000000000000 0 -0.0300000000000000 0 0 0 0 -0.0300000000000000 (-0.541429865846855 + y)*(-0.541429865846855 + y) + (-0.541429865846855 + z)*(-0.541429865846855 + z) + x*x 0 0 0 -0.0300000000000000 0 0 0 0 -0.0300000000000000 -0.0300000000000000; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; -0.0300000000000000 -0.0300000000000000 0 0 0 0 -0.0300000000000000 0 0 0 (0.541429865846855 + y)*(0.541429865846855 + y) + (0.541429865846855 + z)*(0.541429865846855 + z) + x*x -0.0300000000000000 0 0 0 0 -0.0300000000000000 0 -0.0300000000000000; 0 0 -0.0300000000000000 0 0 -0.0300000000000000 -0.0300000000000000 0 0 0 -0.0300000000000000 (-0.541429865846855 + y)*(-0.541429865846855 + y) + (0.541429865846855 + x)*(0.541429865846855 + x) + z*z 0 -0.0300000000000000 -0.0300000000000000 0 0 0 0; 0 -0.0300000000000000 0 -0.0300000000000000 0 0 0 -0.0300000000000000 -0.0300000000000000 0 0 0 (0.541429865846855 + x)*(0.541429865846855 + x) + (0.541429865846855 + z)*(0.541429865846855 + z) + y*y 0 -0.0300000000000000 -0.0300000000000000 0 0 0; 0 -0.0300000000000000 0 0 0 0 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 (-1.08285973169371 + z)*(-1.08285973169371 + z) + x*x + y*y 0 0 0 -0.0300000000000000 0; -0.0300000000000000 -0.0300000000000000 -0.0300000000000000 0 0 0 0 0 0 0 0 -0.0300000000000000 -0.0300000000000000 0 (-0.541429865846855 + z)*(-0.541429865846855 + z) + (0.541429865846855 + y)*(0.541429865846855 + y) + x*x 0 0 0 -0.0300000000000000; 0 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 0 0 0 0 -0.0300000000000000 0 0 (1.08285973169371 + y)*(1.08285973169371 + y) + x*x + z*z -0.0300000000000000 0 0; 0 -0.0300000000000000 0 -0.0300000000000000 -0.0300000000000000 0 0 -0.0300000000000000 0 0 -0.0300000000000000 0 0 0 0 -0.0300000000000000 (-0.541429865846855 + z)*(-0.541429865846855 + z) + (0.541429865846855 + x)*(0.541429865846855 + x) + y*y 0 0; 0 0 -0.0300000000000000 0 -0.0300000000000000 -0.0300000000000000 -0.0300000000000000 0 -0.0300000000000000 0 0 0 0 -0.0300000000000000 0 0 0 (0.541429865846855 + x)*(0.541429865846855 + x) + (0.541429865846855 + y)*(0.541429865846855 + y) + z*z 0; 0 0 0 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 -0.0300000000000000 0 0 0 -0.0300000000000000 0 0 0 (1.08285973169371 + x)*(1.08285973169371 + x) + y*y + z*z])[1:s]

@doc """
    Cu(x,y,z,s)

A k-point independent EPM for Cu.
"""
Cu(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.83955536572772 + x)*(-1.83955536572772 + x) + y*y + z*z 0 0 0.282000000000000 0 0.282000000000000 0 0 0 0 0 0 0 0.282000000000000 0 0.282000000000000 0 0 0; 0 (-0.919777682863858 + x)*(-0.919777682863858 + x) + (-0.919777682863858 + y)*(-0.919777682863858 + y) + z*z 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.282000000000000 0; 0 0 (-0.919777682863858 + x)*(-0.919777682863858 + x) + (0.919777682863858 + z)*(0.919777682863858 + z) + y*y 0 0 0 0 0 0 0 0 0 0 0 0 0 0.282000000000000 0 0; 0.282000000000000 0 0 (-1.83955536572772 + y)*(-1.83955536572772 + y) + x*x + z*z 0 0.282000000000000 0 0 0 0 0 0 0 0.282000000000000 0 0 0 0 0.282000000000000; 0 0 0 0 (-0.919777682863858 + y)*(-0.919777682863858 + y) + (0.919777682863858 + z)*(0.919777682863858 + z) + x*x 0 0 0 0 0 0 0 0 0 0.282000000000000 0 0 0 0; 0.282000000000000 0 0 0.282000000000000 0 (1.83955536572772 + z)*(1.83955536572772 + z) + x*x + y*y 0 0 0 0 0 0 0 0 0 0.282000000000000 0 0 0.282000000000000; 0 0 0 0 0 0 (-0.919777682863858 + x)*(-0.919777682863858 + x) + (-0.919777682863858 + z)*(-0.919777682863858 + z) + y*y 0 0 0 0 0 0.282000000000000 0 0 0 0 0 0; 0 0 0 0 0 0 0 (-0.919777682863858 + x)*(-0.919777682863858 + x) + (0.919777682863858 + y)*(0.919777682863858 + y) + z*z 0 0 0 0.282000000000000 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 (-0.919777682863858 + y)*(-0.919777682863858 + y) + (-0.919777682863858 + z)*(-0.919777682863858 + z) + x*x 0 0.282000000000000 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0.282000000000000 0 (0.919777682863858 + y)*(0.919777682863858 + y) + (0.919777682863858 + z)*(0.919777682863858 + z) + x*x 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0.282000000000000 0 0 0 (-0.919777682863858 + y)*(-0.919777682863858 + y) + (0.919777682863858 + x)*(0.919777682863858 + x) + z*z 0 0 0 0 0 0 0; 0 0 0 0 0 0 0.282000000000000 0 0 0 0 0 (0.919777682863858 + x)*(0.919777682863858 + x) + (0.919777682863858 + z)*(0.919777682863858 + z) + y*y 0 0 0 0 0 0; 0.282000000000000 0 0 0.282000000000000 0 0 0 0 0 0 0 0 0 (-1.83955536572772 + z)*(-1.83955536572772 + z) + x*x + y*y 0 0.282000000000000 0 0 0.282000000000000; 0 0 0 0 0.282000000000000 0 0 0 0 0 0 0 0 0 (-0.919777682863858 + z)*(-0.919777682863858 + z) + (0.919777682863858 + y)*(0.919777682863858 + y) + x*x 0 0 0 0; 0.282000000000000 0 0 0 0 0.282000000000000 0 0 0 0 0 0 0 0.282000000000000 0 (1.83955536572772 + y)*(1.83955536572772 + y) + x*x + z*z 0 0 0.282000000000000; 0 0 0.282000000000000 0 0 0 0 0 0 0 0 0 0 0 0 0 (-0.919777682863858 + z)*(-0.919777682863858 + z) + (0.919777682863858 + x)*(0.919777682863858 + x) + y*y 0 0; 0 0.282000000000000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (0.919777682863858 + x)*(0.919777682863858 + x) + (0.919777682863858 + y)*(0.919777682863858 + y) + z*z 0; 0 0 0 0.282000000000000 0 0.282000000000000 0 0 0 0 0 0 0 0.282000000000000 0 0.282000000000000 0 0 (1.83955536572772 + x)*(1.83955536572772 + x) + y*y + z*z])[1:s]

@doc """
    In(x,y,z,s)

A k-point independent EPM for In.
"""
In(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.02232107178321 + x)*(-1.02232107178321 + x) + (-1.02232107178321 + y)*(-1.02232107178321 + y) + z*z 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0200000000000000; 0 (-1.02232107178321 + x)*(-1.02232107178321 + x) + (0.672228496082037 + z)*(0.672228496082037 + z) + y*y 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 (-1.02232107178321 + y)*(-1.02232107178321 + y) + (0.672228496082037 + z)*(0.672228496082037 + z) + x*x 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 (1.34445699216407 + z)*(1.34445699216407 + z) + x*x + y*y 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 (-1.02232107178321 + x)*(-1.02232107178321 + x) + (-0.672228496082037 + z)*(-0.672228496082037 + z) + y*y 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 (-1.02232107178321 + x)*(-1.02232107178321 + x) + (1.02232107178321 + y)*(1.02232107178321 + y) + z*z 0 0 0 0.0200000000000000 0 0 0 0 0; 0 0 0 0 0 0 (-1.02232107178321 + y)*(-1.02232107178321 + y) + (-0.672228496082037 + z)*(-0.672228496082037 + z) + x*x 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 (0.672228496082037 + z)*(0.672228496082037 + z) + (1.02232107178321 + y)*(1.02232107178321 + y) + x*x 0 0 0 0 0 0; 0 0 0 0 0 0.0200000000000000 0 0 0 (-1.02232107178321 + y)*(-1.02232107178321 + y) + (1.02232107178321 + x)*(1.02232107178321 + x) + z*z 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 (0.672228496082037 + z)*(0.672228496082037 + z) + (1.02232107178321 + x)*(1.02232107178321 + x) + y*y 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 (-1.34445699216407 + z)*(-1.34445699216407 + z) + x*x + y*y 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 (-0.672228496082037 + z)*(-0.672228496082037 + z) + (1.02232107178321 + y)*(1.02232107178321 + y) + x*x 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 (-0.672228496082037 + z)*(-0.672228496082037 + z) + (1.02232107178321 + x)*(1.02232107178321 + x) + y*y 0; 0.0200000000000000 0 0 0 0 0 0 0 0 0 0 0 0 0 (1.02232107178321 + x)*(1.02232107178321 + x) + (1.02232107178321 + y)*(1.02232107178321 + y) + z*z])[1:s]

@doc """
    K(x,y,z,s)

A k-point independent EPM for K.
"""
K(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.24808766095835 + x)*(-1.24808766095835 + x) + y*y + z*z 0 0 -0.00900000000000000 0.00750000000000000 -0.00900000000000000 0 0 0.00750000000000000 0 0.00750000000000000 0 0 -0.00900000000000000 0.00750000000000000 -0.00900000000000000 0 0 0; 0 (-0.624043830479176 + x)*(-0.624043830479176 + x) + (-0.624043830479176 + y)*(-0.624043830479176 + y) + z*z 0 0 0 0.00750000000000000 0 0 0 0 0.00750000000000000 0 0.00750000000000000 0.00750000000000000 0.00750000000000000 0 0.00750000000000000 -0.00900000000000000 0; 0 0 (-0.624043830479176 + x)*(-0.624043830479176 + x) + (0.624043830479176 + z)*(0.624043830479176 + z) + y*y 0.00750000000000000 0 0 0 0 0.00750000000000000 0 0 0.00750000000000000 0 0 0.00750000000000000 0.00750000000000000 -0.00900000000000000 0.00750000000000000 0; -0.00900000000000000 0 0.00750000000000000 (-1.24808766095835 + y)*(-1.24808766095835 + y) + x*x + z*z 0 -0.00900000000000000 0.00750000000000000 0 0 0 0 0 0.00750000000000000 -0.00900000000000000 0 0 0.00750000000000000 0 -0.00900000000000000; 0.00750000000000000 0 0 0 (-0.624043830479176 + y)*(-0.624043830479176 + y) + (0.624043830479176 + z)*(0.624043830479176 + z) + x*x 0 0.00750000000000000 0.00750000000000000 0 0 0 0 0 0 -0.00900000000000000 0 0.00750000000000000 0.00750000000000000 0.00750000000000000; -0.00900000000000000 0.00750000000000000 0 -0.00900000000000000 0 (1.24808766095835 + z)*(1.24808766095835 + z) + x*x + y*y 0 0.00750000000000000 0 0 0 0.00750000000000000 0 0 0 -0.00900000000000000 0 0.00750000000000000 -0.00900000000000000; 0 0 0 0.00750000000000000 0.00750000000000000 0 (-0.624043830479176 + x)*(-0.624043830479176 + x) + (-0.624043830479176 + z)*(-0.624043830479176 + z) + y*y 0 0 0 0.00750000000000000 0.00750000000000000 -0.00900000000000000 0 0 0.00750000000000000 0 0.00750000000000000 0; 0 0 0 0 0.00750000000000000 0.00750000000000000 0 (-0.624043830479176 + x)*(-0.624043830479176 + x) + (0.624043830479176 + y)*(0.624043830479176 + y) + z*z 0.00750000000000000 0 0 -0.00900000000000000 0.00750000000000000 0.00750000000000000 0 0 0.00750000000000000 0 0; 0.00750000000000000 0 0.00750000000000000 0 0 0 0 0.00750000000000000 (-0.624043830479176 + y)*(-0.624043830479176 + y) + (-0.624043830479176 + z)*(-0.624043830479176 + z) + x*x 0 -0.00900000000000000 0 0.00750000000000000 0 0 0 0 0.00750000000000000 0.00750000000000000; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; 0.00750000000000000 0.00750000000000000 0 0 0 0 0.00750000000000000 0 -0.00900000000000000 0 (0.624043830479176 + y)*(0.624043830479176 + y) + (0.624043830479176 + z)*(0.624043830479176 + z) + x*x 0.00750000000000000 0 0 0 0 0.00750000000000000 0 0.00750000000000000; 0 0 0.00750000000000000 0 0 0.00750000000000000 0.00750000000000000 -0.00900000000000000 0 0 0.00750000000000000 (-0.624043830479176 + y)*(-0.624043830479176 + y) + (0.624043830479176 + x)*(0.624043830479176 + x) + z*z 0 0.00750000000000000 0.00750000000000000 0 0 0 0; 0 0.00750000000000000 0 0.00750000000000000 0 0 -0.00900000000000000 0.00750000000000000 0.00750000000000000 0 0 0 (0.624043830479176 + x)*(0.624043830479176 + x) + (0.624043830479176 + z)*(0.624043830479176 + z) + y*y 0 0.00750000000000000 0.00750000000000000 0 0 0; -0.00900000000000000 0.00750000000000000 0 -0.00900000000000000 0 0 0 0.00750000000000000 0 0 0 0.00750000000000000 0 (-1.24808766095835 + z)*(-1.24808766095835 + z) + x*x + y*y 0 -0.00900000000000000 0 0.00750000000000000 -0.00900000000000000; 0.00750000000000000 0.00750000000000000 0.00750000000000000 0 -0.00900000000000000 0 0 0 0 0 0 0.00750000000000000 0.00750000000000000 0 (-0.624043830479176 + z)*(-0.624043830479176 + z) + (0.624043830479176 + y)*(0.624043830479176 + y) + x*x 0 0 0 0.00750000000000000; -0.00900000000000000 0 0.00750000000000000 0 0 -0.00900000000000000 0.00750000000000000 0 0 0 0 0 0.00750000000000000 -0.00900000000000000 0 (1.24808766095835 + y)*(1.24808766095835 + y) + x*x + z*z 0.00750000000000000 0 -0.00900000000000000; 0 0.00750000000000000 -0.00900000000000000 0.00750000000000000 0.00750000000000000 0 0 0.00750000000000000 0 0 0.00750000000000000 0 0 0 0 0.00750000000000000 (-0.624043830479176 + z)*(-0.624043830479176 + z) + (0.624043830479176 + x)*(0.624043830479176 + x) + y*y 0 0; 0 -0.00900000000000000 0.00750000000000000 0 0.00750000000000000 0.00750000000000000 0.00750000000000000 0 0.00750000000000000 0 0 0 0 0.00750000000000000 0 0 0 (0.624043830479176 + x)*(0.624043830479176 + x) + (0.624043830479176 + y)*(0.624043830479176 + y) + z*z 0; 0 0 0 -0.00900000000000000 0.00750000000000000 -0.00900000000000000 0 0 0.00750000000000000 0 0.00750000000000000 0 0 -0.00900000000000000 0.00750000000000000 -0.00900000000000000 0 0 (1.24808766095835 + x)*(1.24808766095835 + x) + y*y + z*z])[1:s]

@doc """
    Li(x,y,z,s)

A k-point independent EPM for Li.
"""
Li(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.89455149547847 + x)*(-1.89455149547847 + x) + y*y + z*z 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0; 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (-0.947275747739237 + y)*(-0.947275747739237 + y) + z*z 0 0 0 0.110000000000000 0 0 0 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0.110000000000000 0 0.110000000000000 0 0; 0 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (0.947275747739237 + z)*(0.947275747739237 + z) + y*y 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0.110000000000000 0 0 0.110000000000000 0.110000000000000 0 0.110000000000000 0; 0 0 0.110000000000000 (-1.89455149547847 + y)*(-1.89455149547847 + y) + x*x + z*z 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0; 0.110000000000000 0 0 0 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (0.947275747739237 + z)*(0.947275747739237 + z) + x*x 0 0.110000000000000 0.110000000000000 0 0 0 0 0 0 0 0 0.110000000000000 0.110000000000000 0.110000000000000; 0 0.110000000000000 0 0 0 (1.89455149547847 + z)*(1.89455149547847 + z) + x*x + y*y 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0; 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (-0.947275747739237 + z)*(-0.947275747739237 + z) + y*y 0 0 0 0.110000000000000 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0; 0 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (0.947275747739237 + y)*(0.947275747739237 + y) + z*z 0.110000000000000 0 0 0 0.110000000000000 0.110000000000000 0 0 0.110000000000000 0 0; 0.110000000000000 0 0.110000000000000 0 0 0 0 0.110000000000000 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (-0.947275747739237 + z)*(-0.947275747739237 + z) + x*x 0 0 0 0.110000000000000 0 0 0 0 0.110000000000000 0.110000000000000; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; 0.110000000000000 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0 (0.947275747739237 + y)*(0.947275747739237 + y) + (0.947275747739237 + z)*(0.947275747739237 + z) + x*x 0.110000000000000 0 0 0 0 0.110000000000000 0 0.110000000000000; 0 0 0.110000000000000 0 0 0.110000000000000 0.110000000000000 0 0 0 0.110000000000000 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (0.947275747739237 + x)*(0.947275747739237 + x) + z*z 0 0.110000000000000 0.110000000000000 0 0 0 0; 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0.110000000000000 0 0 0 (0.947275747739237 + x)*(0.947275747739237 + x) + (0.947275747739237 + z)*(0.947275747739237 + z) + y*y 0 0.110000000000000 0.110000000000000 0 0 0; 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 (-1.89455149547847 + z)*(-1.89455149547847 + z) + x*x + y*y 0 0 0 0.110000000000000 0; 0.110000000000000 0.110000000000000 0.110000000000000 0 0 0 0 0 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + z)*(-0.947275747739237 + z) + (0.947275747739237 + y)*(0.947275747739237 + y) + x*x 0 0 0 0.110000000000000; 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 (1.89455149547847 + y)*(1.89455149547847 + y) + x*x + z*z 0.110000000000000 0 0; 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0 0 0.110000000000000 0 0 0.110000000000000 0 0 0 0 0.110000000000000 (-0.947275747739237 + z)*(-0.947275747739237 + z) + (0.947275747739237 + x)*(0.947275747739237 + x) + y*y 0 0; 0 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0.110000000000000 0 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0 (0.947275747739237 + x)*(0.947275747739237 + x) + (0.947275747739237 + y)*(0.947275747739237 + y) + z*z 0; 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 (1.89455149547847 + x)*(1.89455149547847 + x) + y*y + z*z])[1:s]

@doc """
    Na(x,y,z,s)

A k-point independent EPM for Na.
"""
Na(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.89455149547847 + x)*(-1.89455149547847 + x) + y*y + z*z 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0; 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (-0.947275747739237 + y)*(-0.947275747739237 + y) + z*z 0 0 0 0.110000000000000 0 0 0 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0.110000000000000 0 0.110000000000000 0 0; 0 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (0.947275747739237 + z)*(0.947275747739237 + z) + y*y 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0.110000000000000 0 0 0.110000000000000 0.110000000000000 0 0.110000000000000 0; 0 0 0.110000000000000 (-1.89455149547847 + y)*(-1.89455149547847 + y) + x*x + z*z 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0; 0.110000000000000 0 0 0 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (0.947275747739237 + z)*(0.947275747739237 + z) + x*x 0 0.110000000000000 0.110000000000000 0 0 0 0 0 0 0 0 0.110000000000000 0.110000000000000 0.110000000000000; 0 0.110000000000000 0 0 0 (1.89455149547847 + z)*(1.89455149547847 + z) + x*x + y*y 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0; 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (-0.947275747739237 + z)*(-0.947275747739237 + z) + y*y 0 0 0 0.110000000000000 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0; 0 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + x)*(-0.947275747739237 + x) + (0.947275747739237 + y)*(0.947275747739237 + y) + z*z 0.110000000000000 0 0 0 0.110000000000000 0.110000000000000 0 0 0.110000000000000 0 0; 0.110000000000000 0 0.110000000000000 0 0 0 0 0.110000000000000 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (-0.947275747739237 + z)*(-0.947275747739237 + z) + x*x 0 0 0 0.110000000000000 0 0 0 0 0.110000000000000 0.110000000000000; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; 0.110000000000000 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0 (0.947275747739237 + y)*(0.947275747739237 + y) + (0.947275747739237 + z)*(0.947275747739237 + z) + x*x 0.110000000000000 0 0 0 0 0.110000000000000 0 0.110000000000000; 0 0 0.110000000000000 0 0 0.110000000000000 0.110000000000000 0 0 0 0.110000000000000 (-0.947275747739237 + y)*(-0.947275747739237 + y) + (0.947275747739237 + x)*(0.947275747739237 + x) + z*z 0 0.110000000000000 0.110000000000000 0 0 0 0; 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0.110000000000000 0 0 0 (0.947275747739237 + x)*(0.947275747739237 + x) + (0.947275747739237 + z)*(0.947275747739237 + z) + y*y 0 0.110000000000000 0.110000000000000 0 0 0; 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 (-1.89455149547847 + z)*(-1.89455149547847 + z) + x*x + y*y 0 0 0 0.110000000000000 0; 0.110000000000000 0.110000000000000 0.110000000000000 0 0 0 0 0 0 0 0 0.110000000000000 0.110000000000000 0 (-0.947275747739237 + z)*(-0.947275747739237 + z) + (0.947275747739237 + y)*(0.947275747739237 + y) + x*x 0 0 0 0.110000000000000; 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 0 0 0.110000000000000 0 0 (1.89455149547847 + y)*(1.89455149547847 + y) + x*x + z*z 0.110000000000000 0 0; 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0 0 0.110000000000000 0 0 0.110000000000000 0 0 0 0 0.110000000000000 (-0.947275747739237 + z)*(-0.947275747739237 + z) + (0.947275747739237 + x)*(0.947275747739237 + x) + y*y 0 0; 0 0 0.110000000000000 0 0.110000000000000 0.110000000000000 0.110000000000000 0 0.110000000000000 0 0 0 0 0.110000000000000 0 0 0 (0.947275747739237 + x)*(0.947275747739237 + x) + (0.947275747739237 + y)*(0.947275747739237 + y) + z*z 0; 0 0 0 0 0.110000000000000 0 0 0 0.110000000000000 0 0.110000000000000 0 0 0 0.110000000000000 0 0 0 (1.89455149547847 + x)*(1.89455149547847 + x) + y*y + z*z])[1:s]

@doc """
    Pb(x,y,z,s)

A k-point independent EPM for Pb.
"""
Pb(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-0.6715890106758 + x)*(-0.6715890106758 + x) + (-0.6715890106758 + y)*(-0.6715890106758 + y) + (-0.6715890106758 + z)*(-0.6715890106758 + z) -0.0840000000000000 -0.0840000000000000 0 -0.0840000000000000 0 0 -0.0840000000000000 0 0 0 0 0 0 -0.0390000000000000; -0.0840000000000000 (-1.3431780213516 + x)*(-1.3431780213516 + x) + y*y + z*z 0 -0.0840000000000000 0 -0.0840000000000000 0 0 -0.0840000000000000 0 0 0 0 0 0; -0.0840000000000000 0 (-1.3431780213516 + y)*(-1.3431780213516 + y) + x*x + z*z -0.0840000000000000 0 0 -0.0840000000000000 0 0 -0.0840000000000000 0 0 0 0 0; 0 -0.0840000000000000 -0.0840000000000000 (-0.6715890106758 + x)*(-0.6715890106758 + x) + (-0.6715890106758 + y)*(-0.6715890106758 + y) + (0.6715890106758 + z)*(0.6715890106758 + z) 0 0 0 -0.0840000000000000 0 0 -0.0840000000000000 -0.0390000000000000 0 0 0; -0.0840000000000000 0 0 0 (-1.3431780213516 + z)*(-1.3431780213516 + z) + x*x + y*y -0.0840000000000000 -0.0840000000000000 0 0 0 0 -0.0840000000000000 0 0 0; 0 -0.0840000000000000 0 0 -0.0840000000000000 (-0.6715890106758 + x)*(-0.6715890106758 + x) + (-0.6715890106758 + z)*(-0.6715890106758 + z) + (0.6715890106758 + y)*(0.6715890106758 + y) 0 -0.0840000000000000 0 -0.0390000000000000 0 0 -0.0840000000000000 0 0; 0 0 -0.0840000000000000 0 -0.0840000000000000 0 (-0.6715890106758 + y)*(-0.6715890106758 + y) + (-0.6715890106758 + z)*(-0.6715890106758 + z) + (0.6715890106758 + x)*(0.6715890106758 + x) -0.0840000000000000 -0.0390000000000000 0 0 0 0 -0.0840000000000000 0; -0.0840000000000000 0 0 -0.0840000000000000 0 -0.0840000000000000 -0.0840000000000000 x*x + y*y + z*z -0.0840000000000000 -0.0840000000000000 0 -0.0840000000000000 0 0 -0.0840000000000000; 0 -0.0840000000000000 0 0 0 0 -0.0390000000000000 -0.0840000000000000 (-0.6715890106758 + x)*(-0.6715890106758 + x) + (0.6715890106758 + y)*(0.6715890106758 + y) + (0.6715890106758 + z)*(0.6715890106758 + z) 0 -0.0840000000000000 0 -0.0840000000000000 0 0; 0 0 -0.0840000000000000 0 0 -0.0390000000000000 0 -0.0840000000000000 0 (-0.6715890106758 + y)*(-0.6715890106758 + y) + (0.6715890106758 + x)*(0.6715890106758 + x) + (0.6715890106758 + z)*(0.6715890106758 + z) -0.0840000000000000 0 0 -0.0840000000000000 0; 0 0 0 -0.0840000000000000 0 0 0 0 -0.0840000000000000 -0.0840000000000000 (1.3431780213516 + z)*(1.3431780213516 + z) + x*x + y*y 0 0 0 -0.0840000000000000; 0 0 0 -0.0390000000000000 -0.0840000000000000 0 0 -0.0840000000000000 0 0 0 (-0.6715890106758 + z)*(-0.6715890106758 + z) + (0.6715890106758 + x)*(0.6715890106758 + x) + (0.6715890106758 + y)*(0.6715890106758 + y) -0.0840000000000000 -0.0840000000000000 0; 0 0 0 0 0 -0.0840000000000000 0 0 -0.0840000000000000 0 0 -0.0840000000000000 (1.3431780213516 + y)*(1.3431780213516 + y) + x*x + z*z 0 -0.0840000000000000; 0 0 0 0 0 0 -0.0840000000000000 0 0 -0.0840000000000000 0 -0.0840000000000000 0 (1.3431780213516 + x)*(1.3431780213516 + x) + y*y + z*z -0.0840000000000000; -0.0390000000000000 0 0 0 0 0 0 -0.0840000000000000 0 0 -0.0840000000000000 0 -0.0840000000000000 -0.0840000000000000 (0.6715890106758 + x)*(0.6715890106758 + x) + (0.6715890106758 + y)*(0.6715890106758 + y) + (0.6715890106758 + z)*(0.6715890106758 + z)])[1:s]

@doc """
    Rb(x,y,z,s)

A k-point independent EPM for Rb.
"""
Rb(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.19066245481464 + x)*(-1.19066245481464 + x) + y*y + z*z 0 0 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 0 0 0; 0 (-0.595331227407319 + x)*(-0.595331227407319 + x) + (-0.595331227407319 + y)*(-0.595331227407319 + y) + z*z 0 0 0 -0.00200000000000000 0 0 0 0 -0.00200000000000000 0 -0.00200000000000000 -0.00200000000000000 -0.00200000000000000 0 -0.00200000000000000 0 0; 0 0 (-0.595331227407319 + x)*(-0.595331227407319 + x) + (0.595331227407319 + z)*(0.595331227407319 + z) + y*y -0.00200000000000000 0 0 0 0 -0.00200000000000000 0 0 -0.00200000000000000 0 0 -0.00200000000000000 -0.00200000000000000 0 -0.00200000000000000 0; 0 0 -0.00200000000000000 (-1.19066245481464 + y)*(-1.19066245481464 + y) + x*x + z*z 0 0 -0.00200000000000000 0 0 0 0 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 0; -0.00200000000000000 0 0 0 (-0.595331227407319 + y)*(-0.595331227407319 + y) + (0.595331227407319 + z)*(0.595331227407319 + z) + x*x 0 -0.00200000000000000 -0.00200000000000000 0 0 0 0 0 0 0 0 -0.00200000000000000 -0.00200000000000000 -0.00200000000000000; 0 -0.00200000000000000 0 0 0 (1.19066245481464 + z)*(1.19066245481464 + z) + x*x + y*y 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 0 0 0 0 -0.00200000000000000 0; 0 0 0 -0.00200000000000000 -0.00200000000000000 0 (-0.595331227407319 + x)*(-0.595331227407319 + x) + (-0.595331227407319 + z)*(-0.595331227407319 + z) + y*y 0 0 0 -0.00200000000000000 -0.00200000000000000 0 0 0 -0.00200000000000000 0 -0.00200000000000000 0; 0 0 0 0 -0.00200000000000000 -0.00200000000000000 0 (-0.595331227407319 + x)*(-0.595331227407319 + x) + (0.595331227407319 + y)*(0.595331227407319 + y) + z*z -0.00200000000000000 0 0 0 -0.00200000000000000 -0.00200000000000000 0 0 -0.00200000000000000 0 0; -0.00200000000000000 0 -0.00200000000000000 0 0 0 0 -0.00200000000000000 (-0.595331227407319 + y)*(-0.595331227407319 + y) + (-0.595331227407319 + z)*(-0.595331227407319 + z) + x*x 0 0 0 -0.00200000000000000 0 0 0 0 -0.00200000000000000 -0.00200000000000000; 0 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0 0; -0.00200000000000000 -0.00200000000000000 0 0 0 0 -0.00200000000000000 0 0 0 (0.595331227407319 + y)*(0.595331227407319 + y) + (0.595331227407319 + z)*(0.595331227407319 + z) + x*x -0.00200000000000000 0 0 0 0 -0.00200000000000000 0 -0.00200000000000000; 0 0 -0.00200000000000000 0 0 -0.00200000000000000 -0.00200000000000000 0 0 0 -0.00200000000000000 (-0.595331227407319 + y)*(-0.595331227407319 + y) + (0.595331227407319 + x)*(0.595331227407319 + x) + z*z 0 -0.00200000000000000 -0.00200000000000000 0 0 0 0; 0 -0.00200000000000000 0 -0.00200000000000000 0 0 0 -0.00200000000000000 -0.00200000000000000 0 0 0 (0.595331227407319 + x)*(0.595331227407319 + x) + (0.595331227407319 + z)*(0.595331227407319 + z) + y*y 0 -0.00200000000000000 -0.00200000000000000 0 0 0; 0 -0.00200000000000000 0 0 0 0 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 (-1.19066245481464 + z)*(-1.19066245481464 + z) + x*x + y*y 0 0 0 -0.00200000000000000 0; -0.00200000000000000 -0.00200000000000000 -0.00200000000000000 0 0 0 0 0 0 0 0 -0.00200000000000000 -0.00200000000000000 0 (-0.595331227407319 + z)*(-0.595331227407319 + z) + (0.595331227407319 + y)*(0.595331227407319 + y) + x*x 0 0 0 -0.00200000000000000; 0 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 0 0 0 0 -0.00200000000000000 0 0 (1.19066245481464 + y)*(1.19066245481464 + y) + x*x + z*z -0.00200000000000000 0 0; 0 -0.00200000000000000 0 -0.00200000000000000 -0.00200000000000000 0 0 -0.00200000000000000 0 0 -0.00200000000000000 0 0 0 0 -0.00200000000000000 (-0.595331227407319 + z)*(-0.595331227407319 + z) + (0.595331227407319 + x)*(0.595331227407319 + x) + y*y 0 0; 0 0 -0.00200000000000000 0 -0.00200000000000000 -0.00200000000000000 -0.00200000000000000 0 -0.00200000000000000 0 0 0 0 -0.00200000000000000 0 0 0 (0.595331227407319 + x)*(0.595331227407319 + x) + (0.595331227407319 + y)*(0.595331227407319 + y) + z*z 0; 0 0 0 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 -0.00200000000000000 0 0 0 -0.00200000000000000 0 0 0 (1.19066245481464 + x)*(1.19066245481464 + x) + y*y + z*z])[1:s]

@doc """
    Sn(x,y,z,s)

A k-point independent EPM for Sn.
"""
Sn(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.14027227570066 + x)*(-1.14027227570066 + x) + y*y + z*z 0 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0 0 0; 0 (-0.570136137850332 + x)*(-0.570136137850332 + x) + (-0.570136137850332 + y)*(-0.570136137850332 + y) + z*z 0 0 0 0 0 0 0 -0.0560000000000000 0 -0.0560000000000000 -0.0560000000000000 0 -0.0560000000000000 0 0; 0 0 (-0.570136137850332 + x)*(-0.570136137850332 + x) + (1.04495090674709 + z)*(1.04495090674709 + z) + y*y -0.0560000000000000 0 0 0 0 0 0 -0.0560000000000000 0 0 -0.0560000000000000 -0.0690000000000000 -0.0560000000000000 0; 0 0 -0.0560000000000000 (-1.14027227570066 + y)*(-1.14027227570066 + y) + (-1.11022302462516e-16 + x)*(-1.11022302462516e-16 + x) + z*z 0 -0.0560000000000000 0 0 0 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0; -0.0560000000000000 0 0 0 (-0.570136137850332 + y)*(-0.570136137850332 + y) + (-1.11022302462516e-16 + x)*(-1.11022302462516e-16 + x) + (1.04495090674709 + z)*(1.04495090674709 + z) 0 -0.0560000000000000 0 0 0 0 0 -0.0690000000000000 0 0 -0.0560000000000000 -0.0560000000000000; 0 0 0 -0.0560000000000000 0 (-1.04495090674709 + z)*(-1.04495090674709 + z) + (-0.570136137850332 + x)*(-0.570136137850332 + x) + y*y 0 0 0 0 -0.0560000000000000 -0.0690000000000000 0 -0.0560000000000000 0 -0.0560000000000000 0; 0 0 0 0 -0.0560000000000000 0 (-0.570136137850332 + x)*(-0.570136137850332 + x) + (0.570136137850332 + y)*(0.570136137850332 + y) + z*z -0.0560000000000000 0 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0; -0.0560000000000000 0 0 0 0 0 -0.0560000000000000 (-1.04495090674709 + z)*(-1.04495090674709 + z) + (-0.570136137850332 + y)*(-0.570136137850332 + y) + x*x 0 -0.0690000000000000 0 0 0 0 0 -0.0560000000000000 -0.0560000000000000; 0 0 0 0 0 0 0 0 x*x + y*y + z*z 0 0 0 0 0 0 0 0; -0.0560000000000000 -0.0560000000000000 0 0 0 0 0 -0.0690000000000000 0 (0.570136137850332 + y)*(0.570136137850332 + y) + (1.04495090674709 + z)*(1.04495090674709 + z) + x*x -0.0560000000000000 0 0 0 0 0 -0.0560000000000000; 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0 0 -0.0560000000000000 (-0.570136137850332 + y)*(-0.570136137850332 + y) + (0.570136137850332 + x)*(0.570136137850332 + x) + z*z 0 -0.0560000000000000 0 0 0 0; 0 -0.0560000000000000 0 -0.0560000000000000 0 -0.0690000000000000 -0.0560000000000000 0 0 0 0 (0.570136137850332 + x)*(0.570136137850332 + x) + (1.04495090674709 + z)*(1.04495090674709 + z) + y*y 0 -0.0560000000000000 0 0 0; -0.0560000000000000 -0.0560000000000000 0 0 -0.0690000000000000 0 0 0 0 0 -0.0560000000000000 0 (-1.04495090674709 + z)*(-1.04495090674709 + z) + (1.11022302462516e-16 + x)*(1.11022302462516e-16 + x) + (0.570136137850332 + y)*(0.570136137850332 + y) 0 0 0 -0.0560000000000000; 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0 0 0 0 -0.0560000000000000 0 (1.11022302462516e-16 + x)*(1.11022302462516e-16 + x) + (1.14027227570066 + y)*(1.14027227570066 + y) + z*z -0.0560000000000000 0 0; 0 -0.0560000000000000 -0.0690000000000000 -0.0560000000000000 0 0 -0.0560000000000000 0 0 0 0 0 0 -0.0560000000000000 (-1.04495090674709 + z)*(-1.04495090674709 + z) + (0.570136137850332 + x)*(0.570136137850332 + x) + y*y 0 0; 0 0 -0.0560000000000000 0 -0.0560000000000000 -0.0560000000000000 0 -0.0560000000000000 0 0 0 0 0 0 0 (0.570136137850332 + x)*(0.570136137850332 + x) + (0.570136137850332 + y)*(0.570136137850332 + y) + z*z 0; 0 0 0 0 -0.0560000000000000 0 0 -0.0560000000000000 0 -0.0560000000000000 0 0 -0.0560000000000000 0 0 0 (1.14027227570066 + x)*(1.14027227570066 + x) + y*y + z*z])[1:s]

@doc """
    Zn(x,y,z,s)

A k-point independent EPM for Zn.
"""
Zn(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = eigvals([(-1.34427002432143 + z)*(-1.34427002432143 + z) + x*x + y*y 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 -0.0220000000000000 0 0 0 0 0 0 0 0 0 0 0; 0.0630000000000000 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (-0.720347649569776 + y)*(-0.720347649569776 + y) + (-0.672135012160716 + z)*(-0.672135012160716 + z) 0.0200000000000000 0.0200000000000000 0.0200000000000000 0 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 -0.0220000000000000 0 0 0 0 0 0 0; 0.0630000000000000 0.0200000000000000 (-1.44069529913955 + y)*(-1.44069529913955 + y) + (-0.672135012160716 + z)*(-0.672135012160716 + z) + x*x 0 0.0200000000000000 0.0200000000000000 0 0 0.0630000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0 -0.0220000000000000 0 0 0 0 0 0; 0.0630000000000000 0.0200000000000000 0 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (-0.672135012160716 + z)*(-0.672135012160716 + z) + (0.720347649569776 + y)*(0.720347649569776 + y) 0.0200000000000000 0 0.0200000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0630000000000000 0 0 0 -0.0220000000000000 0 0 0 0 0; 0 0.0200000000000000 0.0200000000000000 0.0200000000000000 (-0.672135012160716 + z)*(-0.672135012160716 + z) + x*x + y*y 0.0200000000000000 0.0200000000000000 0.0200000000000000 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 -0.0220000000000000 0 0 0 0; 0.0630000000000000 0 0.0200000000000000 0 0.0200000000000000 (-0.720347649569776 + y)*(-0.720347649569776 + y) + (-0.672135012160716 + z)*(-0.672135012160716 + z) + (1.24767872816767 + x)*(1.24767872816767 + x) 0 0.0200000000000000 0 0.0630000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0 0 0 0 -0.0220000000000000 0 0 0; 0.0630000000000000 0 0 0.0200000000000000 0.0200000000000000 0 (-0.672135012160716 + z)*(-0.672135012160716 + z) + (1.44069529913955 + y)*(1.44069529913955 + y) + x*x 0.0200000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0630000000000000 0 0 0 0 0 -0.0220000000000000 0 0; 0.0630000000000000 0 0 0 0.0200000000000000 0.0200000000000000 0.0200000000000000 (-0.672135012160716 + z)*(-0.672135012160716 + z) + (0.720347649569776 + y)*(0.720347649569776 + y) + (1.24767872816767 + x)*(1.24767872816767 + x) 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 0 0 0 0 -0.0220000000000000 0; 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (-0.720347649569776 + y)*(-0.720347649569776 + y) + z*z 0.0200000000000000 0.0200000000000000 0.0200000000000000 0 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 0; 0 0.0630000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0200000000000000 (-1.44069529913955 + y)*(-1.44069529913955 + y) + x*x + z*z 0 0.0200000000000000 0.0200000000000000 0 0 0.0630000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0; 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0630000000000000 0 0.0200000000000000 0 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (0.720347649569776 + y)*(0.720347649569776 + y) + z*z 0.0200000000000000 0 0.0200000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0630000000000000 0 0; -0.0220000000000000 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0.0200000000000000 0.0200000000000000 0.0200000000000000 x*x + y*y + z*z 0.0200000000000000 0.0200000000000000 0.0200000000000000 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 -0.0220000000000000; 0 0 0.0630000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0200000000000000 0 0.0200000000000000 (-0.720347649569776 + y)*(-0.720347649569776 + y) + (1.24767872816767 + x)*(1.24767872816767 + x) + z*z 0 0.0200000000000000 0 0.0630000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0; 0 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0630000000000000 0 0 0.0200000000000000 0.0200000000000000 0 (1.44069529913955 + y)*(1.44069529913955 + y) + x*x + z*z 0.0200000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0630000000000000 0; 0 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 0 0.0200000000000000 0.0200000000000000 0.0200000000000000 (0.720347649569776 + y)*(0.720347649569776 + y) + (1.24767872816767 + x)*(1.24767872816767 + x) + z*z 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0; 0 -0.0220000000000000 0 0 0 0 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (-0.720347649569776 + y)*(-0.720347649569776 + y) + (0.672135012160716 + z)*(0.672135012160716 + z) 0.0200000000000000 0.0200000000000000 0.0200000000000000 0 0 0 0.0630000000000000; 0 0 -0.0220000000000000 0 0 0 0 0 0.0630000000000000 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0200000000000000 (-1.44069529913955 + y)*(-1.44069529913955 + y) + (0.672135012160716 + z)*(0.672135012160716 + z) + x*x 0 0.0200000000000000 0.0200000000000000 0 0 0.0630000000000000; 0 0 0 -0.0220000000000000 0 0 0 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0630000000000000 0 0.0200000000000000 0 (-1.24767872816767 + x)*(-1.24767872816767 + x) + (0.672135012160716 + z)*(0.672135012160716 + z) + (0.720347649569776 + y)*(0.720347649569776 + y) 0.0200000000000000 0 0.0200000000000000 0 0.0630000000000000; 0 0 0 0 -0.0220000000000000 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0.0200000000000000 0.0200000000000000 0.0200000000000000 (0.672135012160716 + z)*(0.672135012160716 + z) + x*x + y*y 0.0200000000000000 0.0200000000000000 0.0200000000000000 0; 0 0 0 0 0 -0.0220000000000000 0 0 0 0.0630000000000000 0 0.0630000000000000 0 0 0.0630000000000000 0 0.0200000000000000 0 0.0200000000000000 (-0.720347649569776 + y)*(-0.720347649569776 + y) + (0.672135012160716 + z)*(0.672135012160716 + z) + (1.24767872816767 + x)*(1.24767872816767 + x) 0 0.0200000000000000 0.0630000000000000; 0 0 0 0 0 0 -0.0220000000000000 0 0 0 0.0630000000000000 0.0630000000000000 0 0 0.0630000000000000 0 0 0.0200000000000000 0.0200000000000000 0 (0.672135012160716 + z)*(0.672135012160716 + z) + (1.44069529913955 + y)*(1.44069529913955 + y) + x*x 0.0200000000000000 0.0630000000000000; 0 0 0 0 0 0 0 -0.0220000000000000 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0 0 0 0.0200000000000000 0.0200000000000000 0.0200000000000000 (0.672135012160716 + z)*(0.672135012160716 + z) + (0.720347649569776 + y)*(0.720347649569776 + y) + (1.24767872816767 + x)*(1.24767872816767 + x) 0.0630000000000000; 0 0 0 0 0 0 0 0 0 0 0 -0.0220000000000000 0 0 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 0 0.0630000000000000 0.0630000000000000 0.0630000000000000 (1.34427002432143 + z)*(1.34427002432143 + z) + x*x + y*y])[1:s]

"""
    free(x,y,z,s)
"""
free(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = 
    sort([(x-1)^2 + y^2 + z^2,
           x^2 + (y-1)^2 + z^2,
           x^2 + y^2 + (z-1)^2,
           x^2 + y^2 + z^2,
           x^2 + y^2 + (z+1)^2,
           x^2 + (y+1)^2 + z^2,
           (x+1)^2 + y^2 + z^2])[1:s]
"""
    free_fl(m)
The exact Fermi level for a free electron model with `m` electrons.
"""
free_fl(m::Integer)::Real = 1/4*(3*m/π)^(2/3)
"""
    free_be(m)
The exact band energy for free electron model with `m` electrons.
"""
free_be(m::Integer)::Real = 3/40*m^(5/3)*(3/π)^(2/3)


@doc """
A dictionary whose keys are the labels of high symmetry points from the Python
package `seekpath`. The the values are the same labels but in a better-looking
format.
"""
labels_dict=Dict("GAMMA"=>"Γ","X"=>"X","U"=>"U","L"=>"L","W"=>"W","X"=>"X","K"=>"K",
                 "H"=>"H","N"=>"N","P"=>"P","Y"=>"Y","M"=>"M","A"=>"A","L_2"=>"L₂",
                 "V_2"=>"V₂","I_2"=>"I₂","I"=>"I","M_2"=>"M₂","Y"=>"Y",
                 "Z"=>"Z","Z_0"=>"Z₀")

@doc """
    plot_bandstructure(name,basis,rules,expansion_size,sheets,kpoint_dist,
        convention,coordinates)

Plot the band structure of an empirical pseudopotential.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `kpoint_dist::Real`: the distance between k-points in the band plot.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::Int`: the sheets included in the electronic band structure plot.

# Returns
- (`fig::PyPlot.Figure`,`ax::PyCall.PyObject`): the band structure plot
    as a `PyPlot.Figure`.
"""
function plot_bandstructure(epm::Union{epm₋model2D,epm₋model},
    kpoint_dist::Real,expansion_size::Integer;
    func::Union{Nothing,Function}=nothing,sheets::Integer=10)

    sp=pyimport("seekpath")

    basis = [epm.real_latvecs[:,i] for i=1:size(epm.real_latvecs,1)]
    # basis = epm.real_latvecs
    rbasis=epm.recip_latvecs
    atomtypes=epm.atom_types
    atompos=[[0,0,0]]

    # Calculate the energy cutoff of Fourier expansion.
    cutoff=1
    num_terms=0
    rtol=0.2
    atol=10
    while (abs(num_terms - expansion_size) > expansion_size*rtol &&
        abs(num_terms - expansion_size) > atol)
        if num_terms - expansion_size > 0
            cutoff *= 0.95
        else
            cutoff *= 1.1
        end
        num_terms = size(sample_sphere(rbasis,cutoff,[0,0,0]),2)
    end
    
    # Calculate points along symmetry paths using `seekpath` Python package.
    # Currently uses high symmetry paths from the paper: Y. Hinuma, G. Pizzi,
    # Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on
    # crystallography, Comp. Mat. Sci. 128, 140 (2017).
    # DOI: 10.1016/j.commatsci.2016.10.015
    structure=[basis,atompos,atomtypes]
    timereversal=true

    spdict=sp[:get_explicit_k_path](structure,timereversal,kpoint_dist)
    sympath_pts=Array(spdict["explicit_kpoints_abs"]')

    if coordinates == "lattice"
        m=spdict["reciprocal_primitive_lattice"]
        sympath_pts=inv(m)*sympath_pts
    elseif convention == "ordinary"
        sympath_pts=1/(2π).*sympath_pts
    end

    # Determine the x-axis tick positions and labels.
    labels=spdict["explicit_kpoints_labels"]
    sympts_pos = filter(x->x>0,[if labels[i]==""; -1 else i end for i=1:length(labels)])
    λ=spdict["explicit_kpoints_linearcoord"]

    tmp_labels=[labels_dict[l] for l=labels[sympts_pos]]
    tick_labels=tmp_labels
    for i=2:(length(tmp_labels)-1)
        if (sympts_pos[i-1]+1) == sympts_pos[i]
            tick_labels[i]=""
        elseif (sympts_pos[i]+1) == sympts_pos[i+1]
            tick_labels[i]=tmp_labels[i]*"|"*tmp_labels[i+1]
        else
            tick_labels[i]=tmp_labels[i]
        end
    end

    # Eigenvalues in band structure plot
    evals = eval_epm(sympath_pts,epm,sheets=sheets)

    fig,ax=subplots()
    for i=1:epm.sheets ax.scatter(λ,evals[i,:],s=0.1) end
    ax.set_xticklabels(tick_labels)
    ax.set_xticks(λ[sympts_pos])
    ax.grid(axis="x",linestyle="dashed")
    ax.set_xlabel("High symmetry points")
    ax.set_ylabel("Total energy (eV)")
    ax.set_title(epm.name*" band structure plot")
    (fig,ax)
end

@doc """
    plot_bandstructure(name,basis,rules,expansion_size,sheets,kpoint_dist,
        convention,coordinates)
Plot the band structure of an empirical pseudopotential.
# Arguments
- `name`::String: the name of metal.
- `basis::AbstractMatrix{<:Real}`: the lattice vectors of the crystal
    as columns of a 3x3 array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::Int`: the sheets included in the electronic
    band structure plot.
- `kpoint_dist::Real`: the distance between k-points in the band plot.
- `convention::String="angular"`: the convention for going from real to
    reciprocal space. Options include "angular" and "ordinary".
- `coordinates::String="Cartesian"`: the coordinates of the k-points in
    the band structure plot. Options include "Cartesian" and "lattice".
# Returns
- (`fig::PyPlot.Figure`,`ax::PyCall.PyObject`): the band structure plot
    as a `PyPlot.Figure`.
# Examples
```jldoctest
import Pebsi.EPMs: eval_epm,plot_bandstructure
name="Al"
Al_latvecs=[0.0 3.8262 3.8262; 3.8262 0.0 3.8262; 3.8262 3.8262 0.0]
Al_rules=Dict(2.84 => 0.0562,1.42 => 0.0179)
cutoff=100
sheets=10
kpoint_dist=0.001
plot_bandstructure(name,Al_latvecs,Al_rules,cutoff,sheets,kpoint_dist)
# returns
(PyPlot.Figure(PyObject <Figure size 1280x960 with 1 Axes>),
PyObject <AxesSubplot:title={'center':'Al band structure plot'},
xlabel='High symmetry points', ylabel='Total energy (Ry)'>)
"""
function plot_bandstructure(name::String,basis::AbstractMatrix{<:Real},
        rules::Dict{<:Real,<:Real},expansion_size::Integer,
        sheets::Int,kpoint_dist::Real,
        convention::String="angular",coordinates::String="Cartesian";
        func::Union{Nothing,Function}=nothing)

    sp=pyimport("seekpath")

    rbasis=get_recip_latvecs(basis,convention)
    atomtypes=[0]
    atompos=[[0,0,0]]

    # Calculate the energy cutoff of Fourier expansion.
    cutoff=1
    num_terms=0
    tol=0.2
    while abs(num_terms - expansion_size) > expansion_size*tol
        if num_terms - expansion_size > 0
            cutoff *= 0.95
        else
            cutoff *= 1.1
        end
        num_terms = size(sample_sphere(rbasis,cutoff,[0,0,0]),2)
    end

    # Calculate points along symmetry paths using `seekpath` Python package.
    # Currently uses high symmetry paths from the paper: Y. Hinuma, G. Pizzi,
    # Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on
    # crystallography, Comp. Mat. Sci. 128, 140 (2017).
    # DOI: 10.1016/j.commatsci.2016.10.015
    structure=[basis,atompos,atomtypes]
    timereversal=true

    spdict=sp[:get_explicit_k_path](structure,timereversal,kpoint_dist)
    sympath_pts=Array(spdict["explicit_kpoints_abs"]')

    if coordinates == "lattice"
        m=spdict["reciprocal_primitive_lattice"]
        sympath_pts=inv(m)*sympath_pts
    elseif convention == "ordinary"
        sympath_pts=1/(2π).*sympath_pts
    end

    # Determine the x-axis tick positions and labels.
    labels=spdict["explicit_kpoints_labels"];
    sympts_pos = filter(x->x>0,[if labels[i]==""; -1 else i end for i=1:length(labels)])
    λ=spdict["explicit_kpoints_linearcoord"];

    tmp_labels=[labels_dict[l] for l=labels[sympts_pos]]
    tick_labels=tmp_labels
    for i=2:(length(tmp_labels)-1)
        if (sympts_pos[i-1]+1) == sympts_pos[i]
            tick_labels[i]=""
        elseif (sympts_pos[i]+1) == sympts_pos[i+1]
            tick_labels[i]=tmp_labels[i]*"|"*tmp_labels[i+1]
        else
            tick_labels[i]=tmp_labels[i]
        end
    end

    # Eigenvalues in band structure plot
    evals = eval_epm(sympath_pts,rbasis,rules,cutoff,sheets,func=func)

    fig,ax=subplots()
    for i=1:10 ax.scatter(λ,evals[i,:],s=0.1) end
    ax.set_xticklabels(tick_labels)
    ax.set_xticks(λ[sympts_pos])
    ax.grid(axis="x",linestyle="dashed")
    ax.set_xlabel("High symmetry points")
    ax.set_ylabel("Total energy (Ry)")
    ax.set_title(name*" band structure plot")
    (fig,ax)
end

end
