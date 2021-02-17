using Test

import Pebsi.RectangularMethod: sample_unitcell, symreduce_grid,
    calculate_orbits, convert_mixedradix

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
                recip_latvecs,N,grid_offset,pointgroup,rtol,atol)

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
                @test all([contains(orbits[j][:,i],test_orbits[j],rtol,atol)
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
                recip_latvecs,N,grid_offset,pointgroup,rtol,atol)

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
                test = all([contains(orbits[j][:,i],test_orbits[j],rtol,atol)
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
end
