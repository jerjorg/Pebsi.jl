using Test

import Pebsi.RectangularMethod: sample_unitcell, symreduce_grid,
    calculate_orbits, convert_mixedradix, rectangular_method
import Pebsi.EPMs: free,free_fl,free_be, mf, free_epm

import Base.Iterators: product
import SymmetryReduceBZ.Symmetry: calc_pointgroup, calc_ibz, mapto_ibz,
    calc_spacegroup
import SymmetryReduceBZ.Lattices: get_recip_latvecs
import SymmetryReduceBZ.Utilities: contains

@testset "RectangularMethod" begin
    @testset "sample_unitcell" begin
        recip_latvecs = [1 0; 0 1]
        N = [4 0; 0 4]
        grid_offset = [0.5, 0.5]
        grid_pts = sample_unitcell(recip_latvecs,N,grid_offset)
        answer = [0.125 0.375 0.625 0.875 0.125 0.375 0.625 0.875 0.125 #=
            =# 0.375 0.625 0.875 0.125 0.375 0.625 0.875; 0.125 0.125 0.125 #=
            =# 0.125 0.375 0.375 0.375 0.375 0.625 0.625 0.625 0.625 0.875 #=
            =# 0.875 0.875 0.875]
        @test grid_pts == answer
    end

    @testset "symreduce_grid" begin
        rtol=1e-9
        atol=1e-9
        real_latvecs = [1 0; 0 1]
        atom_types = [0]
        atom_pos = Array([0 0]')
        grid_offset = [0.5, 0.5]
        coordinates = "Cartesian"
        ibzformat = "convex hull"
        makeprim = false
        convention = "ordinary"
        include_orbits = true
        N_list = [[3 0; 0 3],[1 0; 0 12],[8 5; 7  3]]
        for N=N_list

            recip_latvecs = get_recip_latvecs(real_latvecs,convention)
            gridpts = sample_unitcell(recip_latvecs,N,grid_offset)
            (ftrans,pointgroup) = calc_spacegroup(real_latvecs,atom_types,
                atom_pos,coordinates)
            (kpoint_weights,unique_kpoints,orbits) = symreduce_grid(
                recip_latvecs,N,grid_offset,pointgroup,rtol=rtol,atol=atol)

            test_orbits = calculate_orbits(gridpts,pointgroup,recip_latvecs)

            pts1 = reduce(hcat,test_orbits)
            pts2 = reduce(hcat, orbits)
            @test size(pts1) == size(pts2)
            @test all([contains(pts1[:,i],pts2) for i=1:size(pts1,2)])
            @test all([contains(pts2[:,i],pts1) for i=1:size(pts2,2)])

            same_size = all([size(x) for x=test_orbits] ==
                [size(x) for x=orbits])
            @test same_size
            for j=1:length(orbits)
                @test all([contains(orbits[j][:,i],test_orbits[j],rtol=rtol,
                    atol=atol)
                    for i=1:size(orbits[j],2)])
            end
        end

        N_list = [[4 0 0; 0 4 0; 0 0 4],[1 0 0; 0 12 0; 0 0 5],[8 5 9; 7 3 6; #=
            =# 5 3 7]]
            rtol=1e-9
            atol=1e-9
            real_latvecs = [1 0 0; 0 1 0; 0 0 1]
            atom_types = [0]
            atom_pos = Array([0 0 0]')
            grid_offset = [0.5,0.5,0.5]
            coordinates = "Cartesian"
            ibzformat = "convex hull"
            makeprim = false
            convention = "ordinary"

        for N=N_list

            recip_latvecs = get_recip_latvecs(real_latvecs,convention)
            gridpts = sample_unitcell(recip_latvecs,N,grid_offset)
            (ftrans,pointgroup) = calc_spacegroup(real_latvecs,atom_types,
                atom_pos,coordinates)
            (kpoint_weights,unique_kpoints,orbits) = symreduce_grid(
                recip_latvecs,N,grid_offset,pointgroup,rtol=rtol,atol=atol)

            test_orbits = calculate_orbits(gridpts,pointgroup,recip_latvecs)

            test1 = reduce(hcat,test_orbits)
            test2 = reduce(hcat, orbits)
            @test size(test1) == size(test2)
            @test all([contains(test1[:,i],test2) for i=1:size(test1,2)])
            @test all([contains(test2[:,i],test1) for i=1:size(test2,2)]);

            same_size = all([size(x) for x=test_orbits] == [size(x)
                for x=orbits])
            @test same_size
            for j=1:length(orbits)
                test = all([contains(orbits[j][:,i],test_orbits[j],rtol=rtol,
                    atol=atol)
                    for i=1:size(orbits[j],2)])
                @test test
            end
        end
    end

    @testset "convert_mixedradix" begin
        (a,c,f) = (3,4,5)
        test = reduce(hcat,[convert_mixedradix([i,j,k],[a,c,f]) for
            (i,j,k)=product(0:a-1,0:c-1,0:f-1)])
        @test test[1:end] == 1:length(test)

        (a,c) = (4,3)
        test = reduce(hcat,[convert_mixedradix([i,j],[a,c]) for
            (i,j)=product(0:a-1,0:c-1)])
        @test test[1:end] == 1:length(test)
    end

    @testset "rectangular_method" begin
        atom_types = [0]
        atom_pos = Array([0 0]')
        coordinates = "Cartesian"
        convention = "ordinary"
        energy_factor = 1
        N = [20 0; 0 20]
        grid_offset = [0.5,0.5]
        sheets=10

        vars1 = ["real_latvecs","recip_latvecs","rules","cutoff"]
        vars2 = ["electrons","fermilevel","bandenergy"]

        for model=1:5,area=1:3

            for var=vars1
                tmp = Symbol("m"*string(model)*var)
                @eval import Pebsi.EPMs: $tmp
            end

            for var=vars2
                tmp = Symbol("m"*string(model)*var*string(area))
                @eval import Pebsi.EPMs: $tmp
            end

            real_latvecs = getfield(Main,Symbol("m"*string(model)*
                "real_latvecs"))
            recip_latvecs = getfield(Main,Symbol("m"*string(model)*
                "recip_latvecs"))
            rules = getfield(Main,Symbol("m"*string(model)*"rules"))
            cutoff = getfield(Main,Symbol("m"*string(model)*"cutoff"))

            bandenergy_sol = getfield(Main,Symbol("m"*string(model)*
                "bandenergy"*string(area)))
            fermilevel_sol = getfield(Main,Symbol("m"*string(model)*
                "fermilevel"*string(area)))
            electrons = getfield(Main,Symbol("m"*string(model)*
                "electrons"*string(area)))
            
            (num_unique,fermilevel,bandenergy) = rectangular_method(
                recip_latvecs,rules,electrons,cutoff,sheets,N,grid_offset,energy_factor)

            @test abs(bandenergy - bandenergy_sol)/bandenergy_sol < 1e-3
        end
        
        rtol=1e-9
        atol=1e-9
        real_latvecs = [1 0 0; 0 1 0; 0 0 1]
        atom_types = [0]
        atom_pos = Array([0 0 0]')
        electrons = 3
        cutoff = 1.0
        sheets = 7
        rules = Dict(0.0 => 0.0)
        grid_offset = [0.5,0.5,0.5]
        convention = "ordinary"
        coordinates = "Cartesian"
        energy_factor = 1
        n = 80
        N = [n 0 0; 0 n 0; 0 0 n]
        tmp = rectangular_method(real_latvecs,rules,electrons,cutoff,sheets,N,
            grid_offset,energy_factor,rtol=rtol,atol=atol,func=free)
            
        fl = free_fl(electrons)
        be = free_be(electrons)

        @test abs(fl-tmp[2]) < 1e-3
        @test abs(be - tmp[3]) < 1e-4
    end

    n,fl,be = rectangular_method(mf,100)
    @test abs(fl - mf.fermilevel) < 1e-3
    @test abs(be - mf.bandenergy) < 1e-7
    
    n,fl,be = rectangular_method(free_epm,50)
    @test abs(fl - free_epm.fermilevel) < 1e-3
    @test abs(be - free_epm.bandenergy) < 1e-6

end
