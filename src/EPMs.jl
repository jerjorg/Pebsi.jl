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

import PyCall: pyimport
import PyPlot: subplots
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

# 2D "toy" empirical pseudopotentials (the form factors are chosen at random)

# Model 1 - square symmetry
convention = "ordinary"
m1real_latvecs = [1 0; 0 1]
m1recip_latvecs = get_recip_latvecs(m1real_latvecs,convention)
m1rules = Dict(1.00 => -0.23, 1.41 => 0.12)
m1electrons1 = 6

m1cutoff = 6.1
m1fermilevel1 = 0.9381309375519309
m1bandenergy1 = 1.0423513259405526

m1electrons2 = 7
m1fermilevel2 = 1.1057574662181007
m1bandenergy2 = 1.5535874027360785

m1electrons3 = 8
m1fermilevel3 = 1.258616280256323
m1bandenergy3 = 2.145219996962875

# Model 2 - hexagonal symmetry
convention = "ordinary"
m2recip_latvecs = [0.5 0.5; 0.8660254037844386 -0.8660254037844386]
m2real_latvecs = get_recip_latvecs(m2recip_latvecs,convention)
m2rules = Dict(1.0 => 0.39, 1.73 => 0.23, 2.0 => -0.11)
m2cutoff = 5.9

m2electrons1 = 5
m2fermilevel1 = 0.06138423898212197
m2bandenergy1 = -0.4302312741509512

m2electrons2 = 7
m2fermilevel2 = 0.9021827685803184
m2bandenergy2 = -0.024599327665460413

m2electrons3 = 8
m2fermilevel3 = 0.9968615721784458
m2bandenergy3 = 0.38884262264868563

# Model 3 - centered rectangular symmetry
convention = "ordinary"
m3recip_latvecs = [0.4338837391175581 1.0; 0.9009688679024191 0.0]
m3real_latvecs = get_recip_latvecs(m3recip_latvecs,convention)
m3rules = Dict(1.0 => -0.27, 1.06 => 0.2, 1.69 => -0.33)
m3cutoff = 5.95

m3electrons1 = 5
m3fermilevel1 = 0.5833433206577795
m3bandenergy1 = 0.0035066586253235665

m3electrons2 = 7
m3fermilevel2 = 0.9911138305912597
m3bandenergy2 = 0.71444638735834

m3electrons3 = 8
m3fermilevel3 = 1.1117071929086504
m3bandenergy3 = 1.1860046687293682

# Model 4 - rectangular symmetry
convention = "ordinary"
m4recip_latvecs = [1 0; 0 2]
m4real_latvecs = get_recip_latvecs(m4recip_latvecs,convention)
m4rules = Dict(1.0 => 0.39, 2.0 => -0.11, 2.24 => 0.11)
m4cutoff = 8.6

m4electrons1 = 6
m4fermilevel1 = 1.9034249381001005
m4bandenergy1 = 5.1056578173306795

m4electrons2 = 7
m4fermilevel2 = 2.2266488438956333
m4bandenergy2 = 7.1685536634386136

m4electrons3 = 8
m4fermilevel3 = 2.551004975931985
m4bandenergy3 = 9.555193758896971

# Model 5 - oblique symmetry
convention = "ordinary"
m5recip_latvecs = [1.0 -0.4; 0.0 1.0392304845413265]
m5real_latvecs = get_recip_latvecs(m5recip_latvecs,convention)
m5rules = Dict(1.0 => 0.42, 1.11 => 0.02, 1.2 => -0.18)
m5cutoff = 6.3

m5electrons1 = 5
m5fermilevel1 = 0.7916464535133585
m5bandenergy1 = 0.47750270146629903

m5electrons2 = 7
m5fermilevel2 = 1.1470444743280181
m5bandenergy2 = 1.4588171623200643

m5electrons3 = 9
m5fermilevel3 = 1.4883816210907215
m5bandenergy3 = 2.8258962133639556

@doc """
    eval_epm(kpoint,rbasis,rules,cutoff,sheets,energy_conversion_factor,rtol,
        atol)

Evaluate an empirical pseudopotential at a k-point.

# Arguments
- `kpoint::AbstractArray{<:Real,1}:` a point at which the EPM is evaluated.
- `rbasis::AbstractArray{<:Real,2}`: the reciprocal lattice basis as columns of
    a 3x3 real array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets::UnitRange{<:Int}`: the positions of eigenenergies returned.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(rbasis))))`: a relative tolerance for
    finite precision comparisons. This is used for identifying points within a
    circle or sphere in the Fourier expansion.
- `atol::Real=0.0`: an absolute tolerance for finite precision comparisons.

# Returns
- `::AbstractArray{<:Real,1}`: a list of eigenenergies

# Examples
```jldoctest
import PEBSI.EPMs: eval_epm
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
function eval_epm(kpoint::AbstractArray{<:Real,1},
    rbasis::AbstractArray{<:Real,2}, rules::Dict{Float64,Float64}, cutoff::Real,
    sheets::UnitRange{<:Int},energy_conversion_factor::Real=RytoeV,
    rtol::Real=sqrt(eps(float(maximum(rbasis)))),
    atol::Real=0.0)::AbstractArray{<:Real,1}

    if length(kpoint) == 3
        rlatpts = sample_sphere(rbasis,cutoff,kpoint,rtol,atol)
    elseif length(kpoint) == 2
        rlatpts = sample_circle(rbasis,cutoff,kpoint,rtol,atol)
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

    eigvals(Symmetric(ham))[sheets]*energy_conversion_factor
end


@doc """
    eval_epm(kpoints,rbasis,rules,cutoff,sheets,energy_conversion_factor,rtol,
        atol)

Evaluate an empirical pseudopotential at each point in an array.

# Arguments
- `kpoints::AbstractArray{<:Real,2}`: an array of k-points as columns of an
    array.
"""
function eval_epm(kpoints::AbstractArray{<:Real,2},
    rbasis::AbstractArray{<:Real,2}, rules::Dict{Float64,Float64}, cutoff::Real,
    sheets::UnitRange{<:Int},energy_conversion_factor::Real=RytoeV,
    rtol::Real=sqrt(eps(float(maximum(rbasis)))),
    atol::Real=0.0)::AbstractArray{<:Real,2}

    mapslices(x->eval_epm(x,rbasis,rules,cutoff,sheets,energy_conversion_factor,
        rtol,atol),kpoints,dims=1)
end

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
- `basis::AbstractArray{<:Real,2}`: the lattice vectors of the crystal
    as columns of a 3x3 array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::UnitRange{<:Int}`: the sheets included in the electronic
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
import PEBSI.EPMs: eval_epm,plot_bandstructure
name="Al"
Al_latvecs=[0.0 3.8262 3.8262; 3.8262 0.0 3.8262; 3.8262 3.8262 0.0]
Al_rules=Dict(2.84 => 0.0562,1.42 => 0.0179)
cutoff=100
sheets=1:10
kpoint_dist=0.001
plot_bandstructure(name,Al_latvecs,Al_rules,cutoff,sheets,kpoint_dist)
# returns
(PyPlot.Figure(PyObject <Figure size 1280x960 with 1 Axes>),
PyObject <AxesSubplot:title={'center':'Al band structure plot'},
xlabel='High symmetry points', ylabel='Total energy (Ry)'>)
"""
function plot_bandstructure(name::String,basis::AbstractArray{<:Real,2},
        rules::Dict{<:Real,<:Real},expansion_size::Integer,
        sheets::UnitRange{<:Int},kpoint_dist::Real,
        convention::String="angular",coordinates::String="Cartesian")

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
    # Currently use high symmetry paths from the paper: Y. Hinuma, G. Pizzi,
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
    evals=[eval_epm(sympath_pts[:,i],rbasis,rules,cutoff,sheets)
            for i=1:size(sympath_pts,2)]
    evals = reduce(hcat,evals)

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
