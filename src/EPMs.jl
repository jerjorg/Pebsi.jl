@doc """Empirical pseudopotential models for testing band energy calculation methods.
Models based on pseudopotential derivation explained in the textbook Solid
State Physics by Grosso and Parravicini.

Pseudopotential form factors taken from The Fitting of Pseudopotentials to
Experimental Data by Cohen and Heine.

Lattice constants from https://periodictable.com.
"""
module EPMs

import SymmetryReduceBZ.Lattices: genlat_FCC, genlat_BCC, genlat_HEX,
    genlat_BCT, get_recip_latvecs
import SymmetryReduceBZ.Symmetry: calc_spacegroup

import PyCall: pyimport
import PyPlot: subplots
import QHull: chull,Chull
import SymmetryReduceBZ.Lattices: get_recip_latvecs
import SymmetryReduceBZ.Utilities: sample_circle, sample_sphere
import LinearAlgebra: norm, Symmetric, eigvals, dot


epm_names = ["Ag","Al","Au","Cs","Cu","In","K","Li","Na","Pb","Rb","Sn","Zn"]

# The lattice types of the EPMs (follows the naming convention
# of High-throughput electronic band structure calculations:
# Challenges and tools by Wahyu Setyawan and Stefano Curtarolo).
Ag_type = "FCC"
Al_type = "FCC"
Au_type = "FCC"
Cs_type = "BCC"
Cu_type = "BCC"
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
Cu_rtype = "FCC"
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
    "HEX" => [0,0,0.5], "BCT₁" => [0.5,0.5,0.5], "BCT₂" => [0.5,0.5,0.5])

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
Cu_latvecs = genlat_BCC(Cu_abc[1])
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
Ag_rules = Dict(1.41 => 0.195,2.82 => 0.121)
Al_rules = Dict(1.42 => 0.0179,2.84 => 0.0562)
Au_rules = Dict(1.41 => 0.252,2.82 => 0.152)
Cs_rules = Dict(1.33 => -0.03)
Cu_rules = Dict(3.19 => 0.18,2.6 => 0.282)
In_rules = Dict(2.89 => 0.02,3.19 => -0.047)
K_rules = Dict(1.77 => -0.009,1.53 => 0.0075)
Li_rules = Dict(2.32 => 0.11)
Na_rules = Dict(1.9 => 0.0158)
Pb_rules = Dict(2.33 => -0.039,1.16 => -0.084)
Rb_rules = Dict(1.46 => -0.002)
Sn_rules = Dict(4.48 => 0.033,1.65 => -0.056,2.38 => -0.069,3.75 => 0.051)
Zn_rules = Dict(1.34 => -0.022,1.59 => 0.063,1.44 => 0.02)

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

# Cutoffs were compared to a cutoff that had at least 10,000 terms in the 
# expansion at the origin. Cutoffs are such that the difference between the 10th
# eigenvalues is less than 1e-9.
# Ag_cutoff = 8.15
# Al_cutoff = 6.8
# Au_cutoff = 8.12
# Cs_cutoff = 3.98
# Cu_cutoff = 9.9
# In_cutoff = 7.1
# K_cutoff = 3.85
# Li_cutoff = 7.46
# Na_cutoff = 4.78
# Pb_cutoff = 5.54
# Rb_cutoff = 2.7
# Sn_cutoff = 9.9
# Zn_cutoff = 5.24

# Cutoffs are chosen so that there are at least 1000 terms in the expansion at
# the origin.
Ag_cutoff = 8.1
Al_cutoff = 8.2
Au_cutoff = 8.2
Cs_cutoff = 4.3
Cu_cutoff = 7.3
In_cutoff = 7.1
K_cutoff = 5.0
Li_cutoff = 7.5
Na_cutoff = 6.2
Pb_cutoff = 6.7
Rb_cutoff = 4.7
Sn_cutoff = 5.5
Zn_cutoff = 6.6

eVtoRy = 0.07349864435130871395
RytoeV = 13.6056931229942343775

Ag_bz = chull([0.0 -0.1295320008808176 -0.06476600044040881; 0.0 -0.1295320008808176 0.06476600044040881; 0.06476600044040881 -0.1295320008808176 0.0; -0.06476600044040881 -0.1295320008808176 0.0; 0.0 0.06476600044040881 0.1295320008808176; 0.06476600044040881 0.0 0.1295320008808176; -0.06476600044040881 0.0 0.1295320008808176; 0.0 -0.06476600044040881 0.1295320008808176; 0.1295320008808176 -0.06476600044040881 0.0; 0.1295320008808176 0.0 -0.06476600044040881; 0.1295320008808176 0.0 0.06476600044040881; 0.1295320008808176 0.06476600044040881 0.0; 0.0 0.1295320008808176 0.06476600044040881; 0.0 0.1295320008808176 -0.06476600044040881; -0.06476600044040881 0.1295320008808176 0.0; 0.06476600044040881 0.1295320008808176 0.0; 0.0 -0.06476600044040881 -0.1295320008808176; -0.06476600044040881 0.0 -0.1295320008808176; 0.06476600044040881 0.0 -0.1295320008808176; 0.0 0.06476600044040881 -0.1295320008808176; -0.1295320008808176 0.06476600044040881 0.0; -0.1295320008808176 0.0 0.06476600044040881; -0.1295320008808176 0.0 -0.06476600044040881; -0.1295320008808176 -0.06476600044040881 0.0])
Ag_ibz = chull([0.0 -0.09714900066061319 -0.09714900066061324; 0.06476600044040881 -0.0647660004404088 -0.06476600044040884; 0.0 0.0 0.0; -2.7755575615628907e-17 -0.1295320008808176 0.0; 0.03238300022020441 -0.1295320008808176 -0.03238300022020445; -1.3877787807814452e-17 -0.1295320008808176 -0.06476600044040885])

Al_bz = chull([-1.3877787807814478e-17 -0.1306779572421724 -0.06533897862108605; -5.993324074442706e-18 -0.1306779572421724 0.06533897862108622; 0.0653389786210861 -0.1306779572421724 5.551115123125789e-17; -0.06533897862108619 -0.13067795724217238 8.32667268468868e-17; 5.803267077649661e-17 0.06533897862108624 0.13067795724217235; 0.06533897862108624 5.740229089018692e-17 0.13067795724217235; -0.06533897862108616 5.740229089018691e-17 0.1306779572421724; 2.964671527455799e-17 -0.06533897862108616 0.1306779572421724; 0.1306779572421724 -0.06533897862108612 2.7755575615628926e-17; 0.13067795724217238 5.551115123125785e-17 -0.06533897862108617; 0.1306779572421724 4.9517827156815146e-17 0.06533897862108619; 0.1306779572421724 0.06533897862108623 0.0; 5.077858692943453e-17 0.1306779572421724 0.06533897862108619; 4.2894123196062756e-17 0.13067795724217238 -0.06533897862108619; -0.06533897862108615 0.1306779572421724 2.7755575615628926e-17; 0.06533897862108623 0.13067795724217243 0.0; 1.3877787807814475e-17 -0.06533897862108609 -0.13067795724217235; -0.0653389786210861 6.938893903907231e-17 -0.13067795724217238; 0.06533897862108616 6.938893903907232e-17 -0.13067795724217238; 4.2263743309753066e-17 0.06533897862108619 -0.13067795724217238; -0.13067795724217238 0.06533897862108617 5.551115123125789e-17; -0.1306779572421724 4.9517827156815146e-17 0.06533897862108622; -0.1306779572421724 5.551115123125782e-17 -0.06533897862108606; -0.1306779572421724 -0.06533897862108616 8.32667268468868e-17])
Al_ibz = chull([2.0816681711721716e-17 -0.09800846793162929 -0.09800846793162925; 0.06533897862108619 -0.06533897862108622 -0.0653389786210862; 0.0 0.0 0.0; 0.0 -0.1306779572421724 0.0; 0.032669489310543094 -0.13067795724217238 -0.0326694893105431; 1.3877787807814475e-17 -0.1306779572421724 -0.06533897862108615])

Au_bz = chull([0.0 -0.1297572242334592 -0.06487861211672959; 0.0 -0.1297572242334592 0.06487861211672959; 0.06487861211672959 -0.1297572242334592 0.0; -0.06487861211672959 -0.1297572242334592 0.0; 0.0 0.06487861211672959 0.1297572242334592; 0.06487861211672959 0.0 0.1297572242334592; -0.06487861211672959 0.0 0.1297572242334592; 0.0 -0.06487861211672959 0.1297572242334592; 0.1297572242334592 -0.06487861211672959 0.0; 0.1297572242334592 0.0 -0.06487861211672959; 0.1297572242334592 0.0 0.06487861211672959; 0.1297572242334592 0.06487861211672959 0.0; 0.0 0.1297572242334592 0.06487861211672959; 0.0 0.1297572242334592 -0.06487861211672959; -0.06487861211672959 0.1297572242334592 0.0; 0.06487861211672959 0.1297572242334592 0.0; 0.0 -0.06487861211672959 -0.1297572242334592; -0.06487861211672959 0.0 -0.1297572242334592; 0.06487861211672959 0.0 -0.1297572242334592; 0.0 0.06487861211672959 -0.1297572242334592; -0.1297572242334592 0.06487861211672959 0.0; -0.1297572242334592 0.0 0.06487861211672959; -0.1297572242334592 0.0 -0.06487861211672959; -0.1297572242334592 -0.06487861211672959 0.0])
Au_ibz = chull([0.0 -0.09731791817509437 -0.09731791817509441; 0.06487861211672959 -0.06487861211672957 -0.06487861211672961; 0.0 0.0 0.0; -2.775557561562893e-17 -0.1297572242334592 0.0; 0.03243930605836478 -0.1297572242334592 -0.032439306058364814; -1.3877787807814466e-17 -0.1297572242334592 -0.06487861211672961])

Cs_bz = chull([6.938893903907223e-17 6.938893903907223e-18 -0.0861712394871088; -0.043085619743554404 0.04308561974355441 -0.043085619743554404; 6.938893903907228e-18 0.08617123948710881 0.0; 0.04308561974355442 -0.04308561974355439 0.04308561974355442; 0.08617123948710882 0.0 0.0; 0.043085619743554404 0.043085619743554404 0.043085619743554404; 0.04308561974355442 0.043085619743554404 -0.043085619743554404; 0.04308561974355446 -0.04308561974355439 -0.0430856197435544; -0.0430856197435544 0.043085619743554404 0.043085619743554404; 2.0816681711721676e-17 1.387778780781445e-17 0.08617123948710881; -0.043085619743554404 -0.04308561974355439 0.04308561974355438; -0.04308561974355439 -0.043085619743554404 -0.043085619743554356; 5.5511151231257784e-17 -0.08617123948710878 6.938893903907223e-18; -0.0861712394871088 6.938893903907225e-18 3.4694469519536126e-18])
Cs_ibz = chull([0.0 0.0 0.0; -6.5951533254568215e-18 0.04308561974355438 -0.04308561974355442; -0.04308561974355437 0.04308561974355436 -0.04308561974355443; -1.3190306650913638e-17 -1.3190306650913637e-17 -0.08617123948710882])

Cu_bz = chull([0.0 -1.387778780781445e-17 -0.14638716477339264; -0.07319358238669635 0.07319358238669631 -0.07319358238669633; 0.0 0.14638716477339264 0.0; 0.07319358238669632 -0.07319358238669632 0.07319358238669632; 0.14638716477339264 0.0 0.0; 0.07319358238669632 0.07319358238669632 0.07319358238669632; 0.07319358238669632 0.07319358238669632 -0.07319358238669632; 0.07319358238669635 -0.07319358238669633 -0.07319358238669633; -0.07319358238669633 0.07319358238669631 0.07319358238669633; -1.3877787807814455e-17 -1.3877787807814455e-17 0.14638716477339267; -0.07319358238669633 -0.07319358238669633 0.07319358238669633; -0.07319358238669633 -0.07319358238669633 -0.07319358238669632; 0.0 -0.14638716477339264 -1.387778780781445e-17; -0.14638716477339267 -2.0816681711721688e-17 0.0])
Cu_ibz = chull([0.0 0.0 0.0; -9.51647738851254e-18 0.07319358238669632 -0.07319358238669632; -0.07319358238669635 0.07319358238669632 -0.07319358238669635; -1.903295477702508e-17 -1.9032954777025093e-17 -0.14638716477339264])

In_bz = chull([-0.04617851911242334 0.11652893288887838 3.1163680579837e-17; -0.08135372600065084 0.08135372600065084 -0.0534942440193434; -0.08135372600065083 0.08135372600065083 0.053494244019343495; -0.11652893288887836 0.046178519112423315 3.810257448374421e-17; 0.08135372600065087 0.08135372600065081 -0.053494244019343426; 0.08135372600065087 0.08135372600065083 0.053494244019343475; 0.04617851911242336 0.11652893288887833 2.205420054809626e-17; 0.1165289328888784 0.04617851911242331 1.5115306644189037e-17; -0.04617851911242329 -0.04617851911242329 0.10698848803868703; -0.04617851911242329 0.04617851911242329 0.10698848803868703; 0.04617851911242329 -0.04617851911242328 0.10698848803868703; 0.04617851911242329 0.046178519112423266 0.10698848803868703; 0.046178519112423246 0.04617851911242319 -0.10698848803868705; 0.046178519112423246 -0.046178519112423246 -0.10698848803868705; -0.046178519112423204 0.0461785191124232 -0.10698848803868703; -0.046178519112423204 -0.046178519112423246 -0.10698848803868703; 0.04617851911242332 -0.11652893288887836 2.205420054809626e-17; 0.08135372600065084 -0.08135372600065083 0.05349424401934349; 0.08135372600065086 -0.08135372600065086 -0.05349424401934347; 0.11652893288887839 -0.046178519112423294 1.5115306644189037e-17; -0.08135372600065083 -0.08135372600065083 0.053494244019343495; -0.08135372600065081 -0.08135372600065084 -0.05349424401934342; -0.046178519112423315 -0.11652893288887836 3.116368057983699e-17; -0.11652893288887836 -0.046178519112423315 3.810257448374421e-17])
In_ibz = chull([-0.04617851911242335 0.04617851911242331 -0.10698848803868703; -4.098343408303978e-17 0.04617851911242331 -0.10698848803868703; -4.098343408303979e-17 0.0 -0.10698848803868703; 0.0 0.11652893288887839 0.0; 0.0 0.0 0.0; -0.08135372600065084 0.08135372600065084 0.0; -0.04617851911242331 0.11652893288887838 0.0; -0.08135372600065087 0.08135372600065084 -0.05349424401934354])

K_bz = chull([-2.775557561562892e-17 0.0 -0.09931966032676169; -0.049659830163380846 0.049659830163380846 -0.049659830163380846; -1.3877787807814463e-17 0.09931966032676169 0.0; 0.049659830163380846 -0.049659830163380846 0.049659830163380846; 0.09931966032676169 0.0 0.0; 0.049659830163380846 0.049659830163380846 0.049659830163380846; 0.04965983016338083 0.049659830163380846 -0.049659830163380846; 0.04965983016338084 -0.04965983016338085 -0.04965983016338085; -0.04965983016338086 0.04965983016338084 0.04965983016338084; -6.93889390390723e-18 -6.93889390390723e-18 0.09931966032676169; -0.04965983016338086 -0.04965983016338086 0.049659830163380866; -0.04965983016338086 -0.04965983016338084 -0.049659830163380866; -1.3877787807814463e-17 -0.09931966032676169 0.0; -0.09931966032676172 1.0408340855860847e-17 0.0])
K_ibz = chull([0.0 0.0 0.0; 0.0 0.049659830163380846 -0.04965983016338083; -0.04965983016338086 0.04965983016338086 -0.04965983016338083; 0.0 0.0 -0.09931966032676169])

Li_bz = chull([0.0 0.0 -0.15076361772377092; -0.07538180886188546 0.07538180886188546 -0.07538180886188546; 0.0 0.15076361772377092 0.0; 0.07538180886188546 -0.07538180886188546 0.07538180886188546; 0.15076361772377092 0.0 0.0; 0.07538180886188546 0.07538180886188546 0.07538180886188546; 0.07538180886188546 0.07538180886188546 -0.07538180886188546; 0.07538180886188546 -0.07538180886188546 -0.07538180886188546; -0.07538180886188546 0.07538180886188546 0.07538180886188546; 0.0 0.0 0.15076361772377092; -0.07538180886188546 -0.07538180886188546 0.07538180886188546; -0.07538180886188546 -0.07538180886188546 -0.07538180886188546; 0.0 -0.15076361772377092 0.0; -0.15076361772377092 0.0 0.0])
Li_ibz = chull([0.0 0.0 0.0; 0.0 0.07538180886188546 -0.07538180886188546; -0.07538180886188546 0.07538180886188546 -0.07538180886188546; 0.0 0.0 -0.15076361772377092])

Na_bz = chull([2.7755575615628895e-17 0.0 -0.12333345666679002; -0.06166672833339498 0.06166672833339501 -0.06166672833339501; 2.7755575615628914e-17 0.12333345666679002 0.0; 0.061666728333394996 -0.06166672833339499 0.061666728333395024; 0.12333345666678999 6.93889390390723e-18 -6.938893903907228e-18; 0.06166672833339501 0.06166672833339501 0.061666728333394996; 0.06166672833339501 0.061666728333395024 -0.061666728333395; 0.06166672833339501 -0.06166672833339499 -0.06166672833339504; -0.06166672833339498 0.06166672833339504 0.06166672833339504; 1.3877787807814457e-17 1.3877787807814457e-17 0.12333345666679002; -0.06166672833339501 -0.06166672833339498 0.06166672833339498; -0.06166672833339501 -0.061666728333394996 -0.061666728333395; 1.3877787807814444e-17 -0.12333345666678999 -2.775557561562889e-17; -0.12333345666679005 6.938893903907225e-18 0.0])
Na_ibz = chull([0.0 0.0 0.0; 0.0 0.061666728333394996 -0.06166672833339501; -0.06166672833339497 0.061666728333394996 -0.061666728333394996; 0.0 0.0 -0.12333345666679002])

Pb_bz = chull([0.0 -0.1068867107752493 -0.05344335538762467; 0.0 -0.1068867107752493 0.05344335538762467; 0.05344335538762467 -0.1068867107752493 0.0; -0.05344335538762467 -0.1068867107752493 0.0; 0.0 0.05344335538762467 0.1068867107752493; 0.05344335538762467 0.0 0.1068867107752493; -0.05344335538762467 0.0 0.1068867107752493; 0.0 -0.05344335538762467 0.1068867107752493; 0.1068867107752493 -0.05344335538762467 0.0; 0.1068867107752493 0.0 -0.05344335538762467; 0.1068867107752493 0.0 0.05344335538762467; 0.1068867107752493 0.05344335538762467 0.0; 0.0 0.1068867107752493 0.05344335538762467; 0.0 0.1068867107752493 -0.05344335538762467; -0.05344335538762467 0.1068867107752493 0.0; 0.05344335538762467 0.1068867107752493 0.0; 0.0 -0.05344335538762467 -0.1068867107752493; -0.05344335538762467 0.0 -0.1068867107752493; 0.05344335538762467 0.0 -0.1068867107752493; 0.0 0.05344335538762467 -0.1068867107752493; -0.1068867107752493 0.05344335538762467 0.0; -0.1068867107752493 0.0 0.05344335538762467; -0.1068867107752493 0.0 -0.05344335538762467; -0.1068867107752493 -0.05344335538762467 0.0])
Pb_ibz = chull([0.0 -0.08016503308143698 -0.08016503308143702; 0.053443355387624666 -0.05344335538762465 -0.05344335538762467; 0.0 0.0 0.0; 0.0 -0.1068867107752493 0.0; 0.026721677693812344 -0.1068867107752493 -0.026721677693812347; 0.0 -0.1068867107752493 -0.05344335538762469])

Rb_bz = chull([-1.3877787807814457e-17 -6.938893903907228e-18 -0.09474990761884006; -0.047374953809420035 0.04737495380942003 -0.04737495380942003; 0.0 0.09474990761884006 0.0; 0.04737495380942003 -0.04737495380942003 0.04737495380942003; 0.09474990761884006 0.0 0.0; 0.04737495380942003 0.04737495380942003 0.04737495380942003; 0.04737495380942003 0.04737495380942003 -0.04737495380942003; 0.04737495380942002 -0.04737495380942003 -0.04737495380942003; -0.047374953809420015 0.04737495380942002 0.04737495380942002; -6.938893903907227e-18 -6.938893903907227e-18 0.09474990761884007; -0.04737495380942002 -0.04737495380942003 0.04737495380942003; -0.04737495380942003 -0.047374953809420035 -0.047374953809420035; -6.938893903907228e-18 -0.09474990761884006 0.0; -0.09474990761884007 0.0 3.4694469519536123e-18])
Rb_ibz = chull([0.0 0.0 0.0; 0.0 0.047374953809420035 -0.04737495380942003; -0.047374953809420035 0.04737495380942002 -0.04737495380942002; 0.0 0.0 -0.09474990761884006])

Sn_bz = chull([2.0816681711721688e-17 1.0408340855860844e-17 0.10790889310412188; -0.04536999228710132 -0.0453699922871013 0.08315455104857893; -0.09073998457420261 0.0 0.05840020899303592; -0.04536999228710132 0.0453699922871013 0.08315455104857893; 0.0 -0.09073998457420261 0.05840020899303592; 0.04536999228710133 -0.04536999228710129 0.08315455104857891; 0.0 0.09073998457420261 0.05840020899303592; 0.04536999228710133 0.04536999228710129 0.08315455104857891; 0.09073998457420263 -6.938893903907228e-18 0.05840020899303593; 0.09073998457420264 -6.938893903907228e-18 -0.0584002089930359; 0.04536999228710132 0.04536999228710132 -0.0831545510485789; 0.0 0.09073998457420261 -0.05840020899303593; 0.04536999228710132 -0.04536999228710132 -0.0831545510485789; 0.0 -0.09073998457420261 -0.05840020899303593; -0.0453699922871013 0.04536999228710133 -0.0831545510485789; -0.0453699922871013 -0.04536999228710133 -0.0831545510485789; -0.0907399845742026 0.0 -0.05840020899303593; 2.0816681711721664e-17 6.9388939039072214e-18 -0.1079088931041219])
Sn_ibz = chull([-0.09073998457420263 1.3877787807814455e-17 0.0; -0.04536999228710131 -0.0453699922871013 0.0; 0.0 0.0 0.0; -0.09073998457420261 1.3877787807814454e-17 0.05840020899303592; 0.0 0.0 0.1079088931041219; -0.04536999228710131 -0.045369992287101306 0.08315455104857891])

Zn_bz = chull([-0.06619141232616481 0.1146468891736583 0.05348680480525456; -0.06619141232616481 0.11464688917365831 -0.05348680480525455; 0.06619141232616478 0.1146468891736583 0.05348680480525456; 0.06619141232616477 0.11464688917365831 -0.05348680480525455; 0.13238282465232962 -4.4326276067538885e-17 -0.05348680480525455; 0.13238282465232964 -4.43262760675389e-17 0.05348680480525456; 0.06619141232616481 -0.11464688917365833 -0.05348680480525455; 0.06619141232616484 -0.11464688917365833 0.05348680480525456; -0.06619141232616477 -0.11464688917365833 0.05348680480525456; -0.06619141232616478 -0.11464688917365835 -0.05348680480525456; -0.13238282465232962 2.4164565425101098e-17 -0.05348680480525455; -0.13238282465232964 4.83291308502022e-17 0.05348680480525456])
Zn_ibz = chull([0.0 0.0 0.05348680480525454; 0.0 0.0 0.0; -0.06619141232616481 0.11464688917365831 0.05348680480525452; -0.0661914123261648 0.1146468891736583 -3.9661999043715395e-17; 1.1674136499448448e-17 0.11464688917365831 1.487324964139329e-17; 1.1674136499448449e-17 0.11464688917365831 0.05348680480525453])

# Band energy and Fermi level solutions and approximate error computed with about 
# 3 million k-points with the rectangular method
Ag_flans = 2.1922390847785764
Al_flans = 11.591037691304805
Au_flans = 0.25853699543371234
Cs_flans = 1.3360115371222165
Cu_flans = 2.299010605345029
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
Cu_flstd = 0.00914908383769056
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
Cu_beans = 0.44869514208694117
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
Cu_bestd = 0.005879409105478692
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
m1rules = Dict(1.00 => -0.23, 1.41 => 0.12)
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
m2rules = Dict(1.0 => 0.39, 1.73 => 0.23, 2.0 => -0.11)
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
m3rules = Dict(1.0 => -0.27, 1.06 => 0.2, 1.69 => -0.33)
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
m4rules = Dict(1.0 => 0.39, 2.0 => -0.11, 2.24 => 0.11)
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
m5rules = Dict(1.0 => 0.42, 1.11 => 0.02, 1.2 => -0.18)
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
    model2D(energy_conv,sheets,real_latvecs,recip_latvecs,bz,ibz,pointgroup,
    frac_trans,rules,cutoff,electrons,fermiarea,fermilevel,bandenergy)

A container for all the information about the 2D empirical pseudopotential model(s).
"""
struct model
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
    
    rules::Dict{Float64,Float64}
    cutoff::Real       
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
vars₀ = ["energy_conv","sheets","atom_types","atom_pos","coordinates","convention"]
vars₁ = ["real_latvecs","recip_latvecs","bz","ibz","pointgroup","frac_trans","rules","cutoff"]
vars₂ = ["electrons","fermiarea","fermilevel","bandenergy"];
v = Dict()
for i=1:5
    [v[var] = (var |> Symbol |> eval) for var=vars₀]
    [v[var] = ("m"*string(i)*var |> Symbol |> eval) for var=vars₁]
    for j=1:3
        [v[var] = ("m"*string(i)*var*string(j) |> Symbol |> eval) for var=vars₂]
        name = "m"*string(i)*string(j)        
        @eval $(Symbol(name)) = model([v[var] for var=[vars₀; vars₁; vars₂]]...)
    end
end

@doc """
    epm₋model()

A container for all the information about the 2D empirical pseudopotential model(s).
"""
struct epm₋model
    energy_conv::Real
    sheets::Int 
    sym_offset::Vector{<:Real}
    atom_types::Vector{<:Int}
    atom_pos::Matrix{<:Real}
    coordinates::String
    convention::String
    
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
    rules::Dict{Float64,Float64}    
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
convention = "ordinary"
vars₀ = ["energy_conv","sheets","offset","atom_types","atom_pos","coordinates","convention"]
vars₁ = ["type","abc","αβγ","latvecs","rtype","rlatvecs","pointgroup","frac_trans",
        "bz","ibz","rules","electrons","cutoff","fermiarea","flans","flstd","beans","bestd"]
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
    Evaluate an empirical pseudopotential model at a k-point or many k-points.

# Arguments
- `m::Union{model,epm₋model}`: an EPM model structure.
- `kpoint::Union{Vector{<:Real},Matrix{<:Real}}`: a kpoint or kpoints as a 
    vector or as columns of a matrix, respectively.
- `rtol=rtol::Real=sqrt(eps(float(maximum(m.recip_latvecs))))`: a relative tolerance.
- `atol::Real=1e-9`: an absolute tolerance.

# Examples
```jldoctest
import Pebsi.EPMs: m11,eval_epm
eval_epm(m11,[0,0])
# output
10-element Vector{Float64}:
 -0.1577295402106862
  0.7384430928842765
  0.8909287017101515
  0.890928701710161
  1.099286788661485
  1.9651120864887792
  2.0354692798814584
  2.0354692798815126
  2.2049666585171774
  3.894975261713816
```
"""
function eval_epm(m::Union{model,epm₋model},
    kpoint::Union{Vector{<:Real},Matrix{<:Real}};
    rtol=rtol::Real=sqrt(eps(float(maximum(m.recip_latvecs)))),
    atol=1e-9)::Union{Vector{<:Real},Matrix{<:Real}}
    eval_epm(kpoint,m.recip_latvecs,m.rules,m.cutoff,m.sheets,m.energy_conv;
        rtol=rtol,atol=atol)
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
                 "V_2"=>"V₂","I_2"=>"I₂","I"=>"I","M_2"=>"M₂","Y"=>"Y")

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
