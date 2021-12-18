 """Empirical pseudopotential models for testing band energy calculation methods.
Models based on pseudopotential derivation explained in the textbook Solid
State Physics by Grosso and Parravicini.

Pseudopotential form factors taken from The Fitting of Pseudopotentials to
Experimental Data by Cohen and Heine.

Lattice constants from https://periodictable.com.
"""
module EPMs

using ..Defaults: def_atol
using SymmetryReduceBZ.Lattices: genlat_FCC, genlat_BCC, genlat_HEX,
    genlat_BCT, get_recip_latvecs
using SymmetryReduceBZ.Symmetry: calc_spacegroup
using SymmetryReduceBZ.Utilities: sample_circle, sample_sphere
using QHull: chull, Chull
using LinearAlgebra: Symmetric, eigvals
using Distances: SqEuclidean, pairwise!

export epm_names, epm_names2D, eval_epm, epm₋model2D, epm₋model, free, free_fl,
    free_be, free2D, free_fl2D, free_be2D, epms, epms2D, RytoeV, eVtoRy,
    free_epm, mf

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
epm_names2D = ["m"*string(i)*string(j) for i=1:5 for j=1:3]

for name in epm_names @eval export $(Symbol(name*"_epm")) end
for name in epm_names2D @eval export $(Symbol(name)) end

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
sym_offset = Dict("SC" => [0.5,0.5,0.5],"BCC" => [0,0,0],"FCC" => [0.5,0.5,0.5],
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
# Distances are for the angular reciprocal-space convention.
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

# The number of electrons for pseudopotential models
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

# The number of sheets included in band energy calculations
Ag_sheets = 1 + 3
Al_sheets = 4 + 2
Au_sheets = 1 + 2
Cs_sheets = 2 + 2
Cu_sheets = 1 + 2
In_sheets = 4 + 2
K_sheets = 2 + 2
Li_sheets = 1 + 2
Na_sheets = 2 + 2
Pb_sheets = 4 + 2
Rb_sheets = 2 + 2
Sn_sheets = 5 + 2 
Zn_sheets = 3 + 2

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
than 1e-12 for all points of a sparse mesh over the IBZ for all eigenvalues 
beneath and slightly above the Fermi level (up to 1 eV above). The mean 
deviation is for 5 different consecutive expansions where the number of terms in
the expansions for all k-points changed.
=#
# Ag_cutoff = 10.47 # 2232 terms, 3 seconds
# Al_cutoff = 10.49 # 2186 terms, 3 seconds 
# Au_cutoff = 10.48 # 2230 terms, 3 seconds
# Cs_cutoff = 5.86 # 2657 terms, 5 seconds
# Cu_cutoff = 12.48 # 2617 terms, 5 seconds
# In_cutoff = 9.79 # 2797  terms, 5 seconds
# K_cutoff = 6.06 # 1916 terms, 2 seconds
# Li_cutoff = 10.85 # 3144 terms, 7 seconds
# Na_cutoff = 7.40 # 1878 terms, 2 seconds
# Pb_cutoff = 8.58 # 2201 terms, 3 seconds
# Rb_cutoff = 4.58 # 976 terms, 1 second
# Sn_cutoff = 0
# Zn_cutoff = 7.59 # 1498 terms, 0.5 seconds

#=
Cutoffs are chosen such that the mean deviation in all eigenvalues is less
than 1e-12 for all points of a sparse mesh over the IBZ for all eigenvalues 
beneath the Fermi level. The mean deviation is for 5 different consecutive
expansions where the number of terms in the expansions for all k-points changed.
=#
# Ag_cutoff = 9.04 
# Al_cutoff = 10.56 
# Au_cutoff = 11.76 
# Cs_cutoff = 6.4 
# Cu_cutoff = 12.92  
# In_cutoff = 4.26 
# K_cutoff = 3.14 
# Li_cutoff = 4.74 
# Na_cutoff = 3.88 
# Pb_cutoff = 3.98 
# Rb_cutoff = 2.98 
# Sn_cutoff = 3.84 
# Zn_cutoff = 3.7 

eVtoRy = 0.07349864435130871395
RytoeV = 13.6056931229942343775

Ag_bz = chull([0.0 -0.8138735647439266 -0.4069367823719631; -2.25895292678115e-17 -0.8138735647439265 0.4069367823719631; 0.4069367823719631 -0.8138735647439266 0.0; -0.40693678237196307 -0.8138735647439265 0.0; -6.776858780343452e-17 0.4069367823719631 0.8138735647439266; 0.4069367823719631 -4.5179058535623017e-17 0.8138735647439266; -0.4069367823719631 -4.517905853562301e-17 0.8138735647439265; -4.517905853562301e-17 -0.4069367823719631 0.8138735647439265; 0.8138735647439265 -0.40693678237196307 0.0; 0.8138735647439265 0.0 -0.40693678237196307; 0.8138735647439265 -2.2589529267811496e-17 0.40693678237196307; 0.8138735647439265 0.40693678237196307 0.0; -6.776858780343451e-17 0.8138735647439265 0.40693678237196307; -4.517905853562301e-17 0.8138735647439265 -0.40693678237196307; -0.4069367823719631 0.8138735647439265 0.0; 0.4069367823719631 0.8138735647439266 0.0; 0.0 -0.40693678237196307 -0.8138735647439265; -0.40693678237196307 0.0 -0.8138735647439265; 0.4069367823719631 0.0 -0.8138735647439266; -2.25895292678115e-17 0.4069367823719631 -0.8138735647439266; -0.8138735647439265 0.4069367823719631 0.0; -0.8138735647439265 -2.25895292678115e-17 0.4069367823719631; -0.8138735647439266 0.0 -0.4069367823719631; -0.8138735647439266 -0.4069367823719631 0.0])
Ag_ibz = chull([0.0 0.0 0.0; 0.4069367823719631 -0.4069367823719631 -0.40693678237196307; 0.0 -0.6104051735579449 -0.6104051735579448; 9.094148220876481e-17 -0.8138735647439265 0.0; 0.2034683911859815 -0.8138735647439264 -0.20346839118598137; 4.547074110438242e-17 -0.8138735647439265 -0.40693678237196307])

Al_bz = chull([-2.7043196271590465e-16 -0.8210738209162598 -0.410536910458129; 9.014398757196787e-17 -0.8210738209162597 0.41053691045812973; 0.410536910458129 -0.8210738209162598 4.440892098500629e-16; -0.41053691045812973 -0.8210738209162599 6.661338147750941e-16; 6.310079130037775e-16 0.41053691045812984 0.8210738209162597; 0.4105369104581301 4.527469748955415e-16 0.8210738209162596; -0.41053691045812896 2.7245899975160513e-16 0.8210738209162599; 4.507199378598411e-16 -0.41053691045812923 0.8210738209162599; 0.8210738209162596 -0.4105369104581292 2.2204460492503114e-16; 0.8210738209162598 3.2594311394716253e-16 -0.4105369104581295; 0.8210738209162598 0.41053691045812984 0.0; 0.8210738209162597 3.6158946880572304e-16 0.4105369104581297; 4.507199378598408e-16 0.8210738209162599 0.4105369104581294; 9.014398757196834e-17 0.8210738209162598 -0.4105369104581295; -0.410536910458129 0.82107382091626 2.2204460492503104e-16; 0.41053691045812973 0.8210738209162599 0.0; -2.704319627159046e-16 -0.41053691045812923 -0.8210738209162598; -0.4105369104581296 2.0116629003448417e-16 -0.8210738209162598; 0.4105369104581294 3.8145426517842045e-16 -0.82107382091626; -9.014398757196833e-17 0.4105369104581297 -0.8210738209162599; -0.8210738209162599 0.41053691045812896 4.44089209850063e-16; -0.8210738209162599 1.0135185178497033e-18 0.4105369104581294; -0.8210738209162597 -3.463283634071056e-17 -0.41053691045812885; -0.8210738209162599 -0.4105369104581296 6.661338147750938e-16])
Al_ibz = chull([0.0 -0.615805365687195 -0.615805365687195; 0.41053691045813 -0.41053691045813 -0.41053691045813; 0.0 0.0 0.0; 0.0 -0.82107382091626 0.0; 0.205268455229065 -0.82107382091626 -0.205268455229065; 0.0 -0.82107382091626 -0.410536910458129])

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
# 10 million unreduced k-points with the rectangular method

Ag_flans = 2.0700441102197304
Al_flans = 11.591136714325893
Au_flans = 0.25865372280261206
Cs_flans = 1.335887802227545
Cu_flans = 5.241245153830319
In_flans = 8.607306104239635
K_flans = 2.0390882529564673
Li_flans = 4.698493950059316
Na_flans = 3.1443246385587678
Pb_flans = 9.446347550711362
Rb_flans = 1.8557657790353757
Sn_flans = 6.422968933781
Zn_flans = 5.939579523931438

Ag_flstd = 0.0001773117369849546
Al_flstd = 0.00030139376062114415
Au_flstd = 0.0002225855399383628
Cs_flstd = 6.184420114697447e-5
Cu_flstd = 0.0003166654725356247
In_flstd = 0.0001621319681479447
K_flstd = 0.00011991121891166969
Li_flstd = 0.00027630100648338627
Na_flstd = 0.00018490607236637915
Pb_flstd = 0.00028688649963170453
Rb_flstd = 0.00010913070400759909
Sn_flstd = 5.732286913436856e-5
Zn_flstd = 0.0002782756788482695

Ag_beans = 1.2019050029539453
Al_beans = 45.99095682221955 
Au_beans = -2.268020550881349
Cs_beans = 0.23238437422703054
Cu_beans = 7.767170577588814
In_beans = 21.770288170594387
K_beans = 0.5946365698589552
Li_beans = 4.792457142075058
Na_beans = 1.7558200532809254
Pb_beans = 27.469077166344267
Rb_beans = 0.4698610691969691
Sn_beans = 10.472037764215331
Zn_beans = 8.61116425333777

Ag_bestd = 1.0501455840924584e-6
Al_bestd = 2.4139393347065886e-6
Au_bestd = 1.5576771879065765e-6
Cs_bestd = 3.2135758491152617e-8
Cu_bestd = 2.399937897437146e-6
In_bestd = 1.8058794728610238e-6
K_bestd = 5.970754369947948e-8
Li_bestd = 4.812113107040412e-7
Na_bestd = 1.7630214471615413e-7
Pb_bestd = 1.4833009814862601e-6
Rb_bestd = 4.717881748893005e-8
Sn_bestd = 1.8027616880406997e-7
Zn_bestd = 7.553667669791108e-7 

Ag_fermiarea = Ag_electrons/2*Ag_bz.volume
Al_fermiarea = Al_electrons/2*Al_bz.volume
Au_fermiarea = Au_electrons/2*Au_bz.volume
Cs_fermiarea = Cs_electrons/2*Cs_bz.volume
Cu_fermiarea = Cu_electrons/2*Cu_bz.volume
In_fermiarea = In_electrons/2*In_bz.volume
K_fermiarea = K_electrons/2*K_bz.volume
Li_fermiarea = Li_electrons/2*Li_bz.volume
Na_fermiarea = Na_electrons/2*Na_bz.volume
Pb_fermiarea = Pb_electrons/2*Pb_bz.volume
Rb_fermiarea = Rb_electrons/2*Rb_bz.volume
Sn_fermiarea = Sn_electrons/2*Sn_bz.volume
Zn_fermiarea = Zn_electrons/2*Zn_bz.volume

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
m1rules = [1.00 => -0.23, 2.00 => 0.12]

# Cutoffs chosen so the mean deviation of the eigenvalues of 5 consecutive 
# expansions was around 1e-12 for a sparse mesh over the IBZ (about 30-40 
# points in the mesh).
m1cutoff = 7.1
m1electrons1 = 6
m1fermiarea1 = m1electrons1/2*m1bz.volume
m1fermilevel1 = 0.9381315758588166
m1bandenergy1 = 2.084702641629632

m1electrons2 = 7
m1fermiarea2 = m1electrons2/2*m1bz.volume
m1fermilevel2 = 1.1057572321905484
m1bandenergy2 = 3.107174803147335

m1electrons3 = 8
m1fermiarea3 = m1electrons3/2*m1bz.volume
m1fermilevel3 = 1.2586162855632768
m1bandenergy3 = 4.290439984369496

# Model 2 - hexagonal symmetry
convention = "ordinary"
m2recip_latvecs = [0.5 0.5; 0.8660254037844386 -0.8660254037844386]
m2real_latvecs = get_recip_latvecs(m2recip_latvecs,convention)
(m2frac_trans,m2pointgroup) = calc_spacegroup(m2real_latvecs,atom_types,atom_pos,
    coordinates)
m2dist_ff = [[1.00,3.00,4.00],[0.39,0.23,-0.11]]
m2rules = [1.0 => 0.39, 3.00 => 0.23, 4.00 => -0.11]
m2cutoff = 8.6

m2electrons1 = 5
m2fermiarea1 = m2electrons1/2*m2bz.volume
m2fermilevel1 = 0.06138421135450979
m2bandenergy1 = -0.8604627275820277

m2electrons2 = 7
m2fermiarea2 = m2electrons2/2*m2bz.volume
m2fermilevel2 = 0.9021825178151658
m2bandenergy2 = -0.04919886993114208

m2electrons3 = 8
m2fermiarea3 = m2electrons3/2*m2bz.volume
m2fermilevel3 = 0.9968616967367666
m2bandenergy3 = 0.777685005191354

# Model 3 - centered rectangular symmetry
convention = "ordinary"
m3recip_latvecs = [0.4338837391175581 1.0; 0.9009688679024191 0.0]
m3real_latvecs = get_recip_latvecs(m3recip_latvecs,convention)
(m3frac_trans,m3pointgroup) = calc_spacegroup(m3real_latvecs,atom_types,atom_pos,
    coordinates)
m3dist_ff = [[1.00,1.13,2.87],[-0.27,0.2,-0.33]]
m3rules = [1.0 => -0.27, 1.13 => 0.2, 2.87 => -0.33]
m3cutoff = 8.3

m3electrons1 = 5
m3fermiarea1 = m3electrons1/2*m3bz.volume
m3fermilevel1 = 0.5833432286151113
m3bandenergy1 = 0.007013285102808784

m3electrons2 = 7
m3fermiarea2 = m3electrons2/2*m3bz.volume
m3fermilevel2 = 0.9911139068433917
m3bandenergy2 = 1.4288927211671212

m3electrons3 = 8
m3fermiarea3 = m3electrons3/2*m3bz.volume
m3fermilevel3 = 1.1117072183198229
m3bandenergy3 = 2.3720092740928416

# Model 4 - rectangular symmetry
convention = "ordinary"
m4recip_latvecs = [1 0; 0 2]
m4real_latvecs = get_recip_latvecs(m4recip_latvecs,convention)
(m4frac_trans,m4pointgroup) = calc_spacegroup(m4real_latvecs,atom_types,atom_pos,
    coordinates)
m4dist_ff = [[1.00,4.00,5.00],[0.39,-0.11,0.11]]
m4rules = [1.0 => 0.39, 4.00 => -0.11, 5.00 => 0.11]
m4cutoff = 10.2

m4electrons1 = 6
m4fermiarea1 = m4electrons1/2*m4bz.volume
m4fermilevel1 = 1.9034253814658975
m4bandenergy1 = 10.211315560262788

m4electrons2 = 7
m4fermiarea2 = m4electrons2/2*m4bz.volume
m4fermilevel2 = 2.226648774176551
m4bandenergy2 = 14.337107257752455

m4electrons3 = 8
m4fermiarea3 = m4electrons3/2*m4bz.volume
m4fermilevel3 = 2.5510047402843696
m4bandenergy3 = 19.11038747240388

# Model 5 - oblique symmetry
convention = "ordinary"
m5recip_latvecs = [1.0 -0.4; 0.0 1.0392304845413265]
m5real_latvecs = get_recip_latvecs(m5recip_latvecs,convention)
(m5frac_trans,m5pointgroup) = calc_spacegroup(m5real_latvecs,atom_types,atom_pos,
    coordinates)
m5dist_ff = [[1.0,1.24,1.44],[0.42,0.02,-0.18]]
m5rules = [1.0 => 0.42, 1.24 => 0.02, 1.44 => -0.18]
m5cutoff = 11.0

m5electrons1 = 5
m5fermiarea1 = m5electrons1/2*m5bz.volume
m5fermilevel1 = 0.7916464458184026
m5bandenergy1 = 0.9550054013607029

m5electrons2 = 7
m5fermiarea2 = m5electrons2/2*m5bz.volume
m5fermilevel2 = 1.1470443875512362
m5bandenergy2 = 2.9176343162680505

m5electrons3 = 9
m5fermiarea3 = m5electrons3/2*m5bz.volume
m5fermilevel3 = 1.4883818861326563
m5bandenergy3 = 5.65179242276863

@doc """
    epm₋model2D(energy_conv, sheets, atom_types, atom_pos, coordinates, convention,
        real_latvecs, recip_latvecs, bz, ibz, pointgroup, frac_trans, dist_ff, rules,
        cutoff, rlat_type, name, electrons, fermiarea, fermilevel, bandenergy)

A container for information about the 2D empirical pseudopotential models (EPM).

# Arguments
- `energy_conv::Real`: A energy conversion factor energy eigenvalues of EPM.
- `sheets::Integer`: the number of sheets or eigenvalues included in computations.
- `atom_types::Vector{<:Integer}`: a list of atom types as integers in the same 
    order as `atom_pos`.
- `atom_pos::Matrix{<:Real}`: the positions of atoms as columns of a matrix.
- `coordinates::String`: the coordinates in which the atoms are specified in 
    `atom_pos`. Options include "Cartesian" and "lattice".
- `convention::String`: the convention for going between real and reciprocal space.
    Options include "ordinary" and "angular".
- `real_latvecs::Matrix{<:Real}`: the real lattice vectors in Cartesian coordinates
    as columns of a matrix.
- `recip_latvecs::Matrix{<:Real}`: the reciprocal lattice vectors in Cartesian
    coorinates as columns of a matrix.
- `bz::Chull{<:Real}`: the Brillouin zone of the EPM as a convex hull object from
    the Julia package QHull. This can be calculated with the Julia package 
    `SymmetryReduceBZ`.
- `ibz::Chull{<:Real}`: the irreducible Brillouin zone of the EPM.
- `pointgroup::Vector{Matrix{Float64}}`: the point group of the real-space crystal.
- `frac_trans::Vector{Vector{Float64}}`: the fractional translations from the space
    group of the crystal.
- `dist_ff::Vector{Vector{Float64}}`: the distances and form factors for the EPM.
    The list array of the nested array contains the distances and the second array
    contains the corresponding pseudopotential form factors.
- `rules::Vector{Pair{Float64, Float64}}`: another format for the distances and 
    form factors for the EPM.
- `cutoff::Real`: the cutoff distance for the Fourier expansion for the EPM. The
   number of terms kept in the Fourier expansion is the same as the number of 
   lattice points for the reciprocal lattice of the EPM that fit within a circle
    of radius `cutoff`.
- `rlat_type::String`: the lattice of type of the reciprocal lattice.

- `name::String`: the name of the EPM.
- `electrons::Real`: the number of electrons for the EPM.
- `fermiarea::Real`: the Fermi area of the EPM or the area of the shadows of the 
    occupied sheets.
- `fermilevel::Real`: the true Fermi level for the EPM.
- `bandenergy::Real`: the true band energy for the EPM.
"""
mutable struct epm₋model2D
    energy_conv::Real
    sheets::Integer 
    atom_types::Vector{<:Integer}
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
    
    name::String
    electrons::Real
    fermiarea::Real
    fermilevel::Real
    bandenergy::Real
end

m1name1 = "m11"; m1name2 = "m12"; m1name3 = "m13"
m2name1 = "m21"; m2name2 = "m22"; m2name3 = "m23"
m3name1 = "m31"; m3name2 = "m32"; m3name3 = "m33"
m4name1 = "m41"; m4name2 = "m42"; m4name3 = "m43"
m5name1 = "m51"; m5name2 = "m52"; m5name3 = "m53"

energy_conv = 1
sheets = 10
atom_types = [0]
atom_pos = Array([0 0;]')
coordinates = "Cartesian" 
convention = "ordinary"
vars₀ = ["energy_conv","sheets","atom_types","atom_pos","coordinates",
        "convention"]
vars₁ = ["real_latvecs","recip_latvecs","bz","ibz","pointgroup","frac_trans",
        "dist_ff","rules","cutoff","rlat_type"]
vars₂ = ["name","electrons","fermiarea","fermilevel","bandenergy"];
v = Dict()
epms2D = []
for i=1:5
    [v[var] = (var |> Symbol |> eval) for var=vars₀]
    [v[var] = ("m"*string(i)*var |> Symbol |> eval) for var=vars₁]
    for j=1:3
        [v[var] = ("m"*string(i)*var*string(j) |> Symbol |> eval) for var=vars₂]
        name = "m"*string(i)*string(j)        
        @eval $(Symbol(name)) = epm₋model2D([v[var] for var=[vars₀; vars₁; vars₂]]...)
        push!(epms2D, @eval $(Symbol(name)))
    end
end

# Free electron model in 2D
"""
    free2D(x,y,s)

The free electron model in 2D for a square lattice.
"""
free2D(x::Real,y::Real,s::Integer)::AbstractVector{<:Real} = 
    sort([(x-1)^2 + y^2,
            x^2 + (y-1)^2,
            (x-1)^2 + (y-1)^2,
            x^2 + y^2,
            (x+1)^2 + (y+1)^2,
            x^2 + (y+1)^2,
            (x+1)^2 + y^2])[1:s]

@doc """
    free_fl2D(m)
The exact Fermi level for a 2D free electron model with `m` electrons.
"""
free_fl2D(m::Integer)::Real = m/(2π)

@doc """
    free_be2D(m)
The exact band energy for a 2D free electron model with `m` electrons.
"""
free_be2D(m::Integer)::Real = m^2/(4π)

sheets = 7
free2Dreal_latvecs = [1.0 0.0; 0.0 1.0]
(free2Dfrac_trans,free2Dpointgroup) = calc_spacegroup(free2Dreal_latvecs,atom_types,
    atom_pos,coordinates)
free2Drecip_latvecs = get_recip_latvecs(free2Dreal_latvecs,convention)
free2Ddist_ff = [[1.00],[0.0]]
free2Drules = [1.0 => 0.0]
free2Dbz = chull([0.5 0.5; 0.5 -0.5; -0.5 -0.5; -0.5 0.5])
free2Dibz = chull([0.0 0.0; 0.5 0.0; 0.5 0.5])

# Cutoffs chosen so the mean deviation of the eigenvalues of 5 consecutive 
# expansions was around 1e-12 for a sparse mesh over the IBZ (about 30-40 
# points in the mesh).
free2Dcutoff = 4.0
free2Delectrons = 5
free2Dfermiarea = free2Delectrons/2*free2Dbz.volume
free2Dfermilevel = free_fl2D(free2Delectrons)
free2Dbandenergy = free_be2D(free2Delectrons)

# Free electron 2D model
mf = epm₋model2D(energy_conv, sheets, atom_types, atom_pos, coordinates, convention,
    free2Dreal_latvecs, free2Drecip_latvecs, free2Dbz, free2Dibz, free2Dpointgroup,
    free2Dfrac_trans, free2Ddist_ff, free2Drules, free2Dcutoff, "square",
    "free electron model", free2Delectrons, free2Dfermiarea, free2Dfermilevel,
    free2Dbandenergy)

@doc """
    epm₋model(energy_conv, sym_offset, atom_types, atom_pos, coordinates, convention,
        sheets, name, lat_type, lat_constants, lat_angles, real_latvecs, rlat_type, 
        recip_latvecs, pointgroup, frac_trans, bz, ibz, dist_ff, rules, electrons, 
        cutoff, fermiarea, fermilevel, fl_error, bandenergy, be_error)

A container for all the information about the empirical pseudopotential models.

- `energy_conv::Real`: A energy conversion factor energy eigenvalues of EPM.
- `sym_offset::Vector{<:Real}`: a symmetry preserving offset for a regular *k*-point
    grid.
- `atom_types::Vector{<:Integer}`: a list of atom types as integers in the same 
    order as `atom_pos`.
- `atom_pos::Matrix{<:Real}`: the positions of atoms as columns of a matrix.
- `coordinates::String`: the coordinates in which the atoms are specified in 
    `atom_pos`. Options include "Cartesian" and "lattice".
- `convention::String`: the convention for going between real and reciprocal space.
    Options include "ordinary" and "angular".
- `sheets::Integer`: the number of sheets or eigenvalues included in computations.
- `name::String`: the name of the EPM.
- `lat_type::String`: the lattice type for the real-space lattice.
- `lat_constants::Vector{<:Real}`: the lattice constants for the real-space lattice
    and the conventional unit cell.
- `lat_angles::Vector{<:Real}`: the lattice angles for the real-space lattice and 
    conventional unit cell.
- `real_latvecs::Matrix{<:Real}`: the real lattice vectors in Cartesian coordinates
    as columns of a matrix.
- `rlat_type::String`: the lattice of type of the reciprocal lattice.
- `recip_latvecs::Matrix{<:Real}`: the reciprocal lattice vectors in Cartesian
    coorinates as columns of a matrix.
- `pointgroup::Vector{Matrix{Float64}}`: the point group of the real-space crystal.
- `frac_trans::Vector{Vector{Float64}}`: the fractional translations from the space
    group of the crystal.
- `bz::Chull{<:Real}`: the Brillouin zone of the EPM as a convex hull object from
    the Julia package QHull. This can be calculated with the Julia package 
    `SymmetryReduceBZ`.
- `ibz::Chull{<:Real}`: the irreducible Brillouin zone of the EPM.
- `dist_ff::Vector{Vector{Float64}}`: the distances and form factors for the EPM.
    The list array of the nested array contains the distances and the second array
    contains the corresponding pseudopotential form factors.
- `rules::Vector{Pair{Float64, Float64}}`: another format for the distances and 
    form factors for the EPM.
- `electrons::Real`: the number of electrons for the EPM.
- `cutoff::Real`: the cutoff distance for the Fourier expansion for the EPM. The
    number of terms kept in the Fourier expansion is the same as the number of 
    lattice points for the reciprocal lattice of the EPM that fit within a circle
    of radius `cutoff`.
- `fermiarea::Real`: the Fermi area of the EPM or the area of the shadows of the 
    occupied sheets.
- `fermilevel::Real`: the true Fermi level for the EPM.
- `fl_error::Real`: the estimated error in the true Fermi level.
- `bandenergy::Real`: the true band energy for the EPM.
- `be_error::Real`: the estimated error in the true band energy.
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
epms = []
for m=epm_names
    offset = sym_offset[eval(Symbol(m,"_rtype"))]
    [v[var] = eval(Symbol(var)) for var=vars₀]
    [v[var] = eval(Symbol(m,"_",var)) for var=vars₁]
    @eval $(Symbol(m,"_epm")) = epm₋model([v[var] for var=[vars₀;vars₁]]...)
    push!(epms, @eval $(Symbol(m,"_epm")))
end

# Free electron model in 3D
"""
    free(x,y,z,s)

The free electron model in 3D for a simple cubic lattice.
"""
free(x::Real,y::Real,z::Real,s::Integer)::AbstractVector{<:Real} = 
    sort([(x-1)^2 + (y-1)^2 + (z-1)^2,
          x^2 + (y-1)^2 + (z-1)^2,
          (x+1)^2 + (y-1)^2 + (z-1)^2,

          (x-1)^2 + y^2 + (z-1)^2,
          x^2 + y^2 + (z-1)^2,
          (x+1)^2 + y^2 + (z-1)^2,

          (x-1)^2 + (y+1)^2 + (z-1)^2,
          x^2 + (y+1)^2 + (z-1)^2,
          (x+1)^2 + (y+1)^2 + (z-1)^2,           
           
          (x-1)^2 + (y-1)^2 + z^2,
          x^2 + (y-1)^2 + z^2,
          (x+1)^2 + (y-1)^2 + z^2,

          (x-1)^2 + y^2 + z^2,
          x^2 + y^2 + z^2,
          (x+1)^2 + y^2 + z^2,

          (x-1)^2 + (y+1)^2 + z^2,
          x^2 + (y+1)^2 + z^2,
          (x+1)^2 + (y+1)^2 + z^2,           

          (x-1)^2 + (y-1)^2 + (z+1)^2,
          x^2 + (y-1)^2 + (z+1)^2,
          (x+1)^2 + (y-1)^2 + (z+1)^2,

          (x-1)^2 + y^2 + (z+1)^2,
          x^2 + y^2 + (z+1)^2,
          (x+1)^2 + y^2 + (z+1)^2,

          (x-1)^2 + (y+1)^2 + (z+1)^2,
          x^2 + (y+1)^2 + (z+1)^2,
          (x+1)^2 + (y+1)^2 + (z+1)^2])[1:s]

"""
    free_fl(m)
The exact Fermi level for a free electron model with `m` electrons.
"""
free_fl(m::Integer)::Real = 1/4*(3*m/π)^(2/3)

"""
    free_be(m)
The exact band energy for free electron model with `m` electrons.
"""
free_be(m::Integer)::Real = 2*3/40*m^(5/3)*(3/π)^(2/3)

energy_conv = 1
freesym_offset = [0.,0.,0.]
convention = "ordinary"
free_lat_type = "SC"
free_rlat_type = "SC"
free_lat_constants = [1.,1.,1.]
free_lat_angles = [π/2, π/2, π/2]
free_real_latvecs = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
free_recip_latvecs = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
(free_frac_trans, free_pointgroup) = calc_spacegroup(free_real_latvecs, atom_types, atom_pos, coordinates)
free_bz = chull([0.5 -0.5 0.5; 0.5 -0.5 -0.5; 0.5 0.5 -0.5; 0.5 0.5 0.5; -0.5 0.5 -0.5; -0.5 0.5 0.5; -0.5 -0.5 -0.5; -0.5 -0.5 0.5])
free_ibz = chull([0.5 0.0 0.0; 0.5 -0.5 0.0; 0.5 -0.5 0.5; 0.0 0.0 0.0])
free_dist_ff = [[1.0],[0.0]]
free_rules = [1.0 => 0.0]
free_electrons = 5
free_cutoff = 4.0
free_fermiarea = free_electrons/2
free_fermilevel = free_fl(free_electrons)
free_bandenergy = free_be(free_electrons)

free_epm = epm₋model(energy_conv, freesym_offset, atom_types, atom_pos, coordinates,
    convention, sheets, "free electron model", free_lat_type, free_lat_constants,
    free_lat_angles, free_real_latvecs, free_rlat_type, free_recip_latvecs,
    free_pointgroup, free_frac_trans, free_bz, free_ibz, free_dist_ff, free_rules,
    free_electrons, free_cutoff, free_fermiarea,free_fermilevel, 0.0,
    free_bandenergy, 0.0)

@doc """
    eval_epm(kpoints,rbasis,rules,cutoff,sheets,energy_conversion_factor;rtol,atol,func)

Evaluate an empirical pseudopotential at each point in an array.

# Arguments
- `kpoints::AbstractMatrix{<:Real}`: an array of k-points as columns of a matrix.
"""
function eval_epm(kpoints::AbstractMatrix{<:Real},
    rbasis::AbstractMatrix{<:Real}, rules, cutoff::Real,
    sheets::Integer, energy_conversion_factor::Real=RytoeV;
    rtol::Real=sqrt(eps(float(maximum(rbasis)))), atol::Real=def_atol,
    func::Union{Nothing,Function}=nothing)::AbstractMatrix{<:Real}

    if !(func == nothing)
        eval_epm(func,kpoints,sheets)
    end

    mapslices(x->eval_epm(x,rbasis,rules,cutoff,sheets,energy_conversion_factor;
        rtol=rtol,atol=atol),kpoints,dims=1)
end

@doc """
    eval_epm(kpoint,rbasis,rules,cutoff,sheets,energy_conversion_factor;rtol,
        atol,func)

Evaluate an empirical pseudopotential model (EPM) at a k-point.

# Arguments
- `kpoint::AbstractVector{<:Real}:` a point at which the EPM is evaluated.
- `rbasis::AbstractMatrix{<:Real}`: the reciprocal lattice vectors in columns of
    a matrix.
- `rules`: a vector of pairs where the first elements of the pairs are distances
    between reciprocal lattice points rounded to two decimals places and second
    elements are the empirical pseudopotential form factors.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets::Integer`: the number of eigenenergies returned.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(rbasis))))`: a relative tolerance for
    finite precision comparisons. This is used for identifying points within a
    circle or sphere in the Fourier expansion.
- `atol::Real=def_atol`: an absolute tolerance for finite precision comparisons.
- `func::Union{Nothing,Function}=nothing`: a k-point independent EPM.

# Returns
- `::AbstractVector{<:Real}`: a list of eigenenergies

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm
kpoint = [0,0,0]
rlatvecs = [1 0 0; 0 1 0; 0 0 1]
rules = [1.00 => .01, 2.00 => 0.015]
cutoff = 3.0
sheets = 10
eval_epm(kpoint, rlatvecs, rules, cutoff, sheets)
# output
 10-element Vector{Float64}:
 -0.025091555116792823
 13.191390044925443
 13.191390044925663
 13.591143909078465
 13.591143909078552
 13.591143909078642
 14.396491921113055
 26.800000569752516
 26.80000056975255
 26.800303409287515
```
"""
function eval_epm(kpoint::AbstractVector{<:Real},
    rbasis::AbstractMatrix{<:Real}, rules, cutoff::Real,
    sheets::Integer, energy_conversion_factor::Real=RytoeV;
    rtol::Real=sqrt(eps(float(maximum(rbasis)))),
    atol::Real=def_atol,
    func::Union{Nothing,Function}=nothing)::AbstractVector{<:Real}

    # extra
    if !(func == nothing)
        return func(kpoint...,sheets)
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
    eigvals(Symmetric(ham))[1:sheets]*energy_conversion_factor
end

@doc """
    eval_epm(kpoint,epm;rtol,atol,func,sheets,func)

Calculate the eigenvalues of an empirical pseudopotential at a *k*-point.

# Arguments
- `kpoint::AbstractVector{<:Real}`: a *k*-point in Cartesian coordinates at 
    which to evaluate the EPM.
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs))))`: a relative tolerance
    for finite-precision comparisons. In this case, the tolerance identifies points
    that are close to being within a sphere or circle in the Fourier expansion.
- `atol::Real=def_atol`: an absolute tolerance for finite-precision comparisons.
- `sheets::Integer=-1`: the number of sheets for the k-point independent EPM. If
    not provided, the number of sheets specified by the EPM container is used.
- `func::Union{Nothing,Function}=nothing`: a *k*-point independent function.

# Returns
- `::AbstractVector{<:Real}`: the eigenvalues at the specified *k*-point for the EPM.

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm,Al_epm
eval_epm([0,0,0],Al_epm)
# output
6-element Vector{Float64}:
 -0.05904720925479707
 26.68476356943067
 26.684763569434374
 26.68476356943802
 26.713073023266258
 28.17466342570331
```
"""
function eval_epm(kpoint::AbstractVector{<:Real},
    epm::Union{epm₋model2D,epm₋model};
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))),
    atol::Real=def_atol, sheets::Integer=0, 
    func::Union{Nothing,Function}=nothing)::AbstractVector{<:Real}

    if sheets == 0 sheets = epm.sheets end
    eval_epm(kpoint, epm.recip_latvecs, epm.rules, epm.cutoff, sheets, epm.energy_conv,
        rtol=rtol, atol=atol, func=func)
end

@doc """
    eval_epm(kpoints,epm,rtol,atol,func,sheets,func)

Evaluate an EPM an many *k*-points.

# Arguments
- `kpoints::AbstractMatrix{<:Real}`: a matrix whose columns are *k*-point points.
- `epm::Union{epm₋model2D,epm₋model}`: an EPM model.
- `rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs))))`: a relative tolerance.
- `atol::Real=def_atol`: an absolute tolerance.
- `sheets::Integer=0`: the number of sheets for the *k*-point independent EPM.
    If not provided, the number of sheets specified by the EPM container is used.
- `func::Union{Nothing,Function}=nothing`: a *k*-point independent function.

# Returns
- `::AbstractMatrix{<:Real}`: a matrix whose columns are eigenvalues of the EPM 
    at the *k*-point in the same column in `kpoints`.

# Examples
```jldoctest
import Pebsi.EPMs: eval_epm, Al_epm
eigvals = eval_epm([0 0; 0 1; 0 0],Al_epm)
size(eigvals)
# output
(6, 2)
```
"""
function eval_epm(kpoints::AbstractMatrix{<:Real},
    epm::Union{epm₋model2D,epm₋model};
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))),
    atol=def_atol,sheets::Integer=0,
    func::Union{Nothing,Function}=nothing)::AbstractMatrix{<:Real}

    if sheets == 0 sheets = epm.sheets end
    mapslices(x->eval_epm(x,epm,rtol=rtol,atol=atol,sheets=sheets,func=func),
        kpoints,dims=1)
end
end # module