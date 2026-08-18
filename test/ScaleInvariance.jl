using Test
using Pebsi.Geometry: sample_simplex, barytocart, order_vertices!, simplex_size
using Pebsi.Polynomials: getpoly_coeffs, eval_poly
using Pebsi.QuadraticIntegration: quad_area_volume
using FastGaussQuadrature: gausslegendre

# A patch's coefficients are barycentric, so the fraction of the triangle lying
# below zero is a property of the coefficients alone. Area is therefore that
# fraction times the triangle's size, and `area/size` is invariant under every
# affine map of the triangle - translation, rotation, uniform scaling and
# anisotropic stretching alike. Volume behaves the same way.
#
# That gives a property to test against instead of a stored number, which matters
# because the failures this file exists to catch are ones where a stored number
# still looks plausible. An absolute tolerance anywhere in the integration path
# breaks this invariant and nothing else, since the geometry it claims to measure
# has not changed.
#
# Absolute correctness is anchored separately, against the reference integrator
# below and against closed forms, so the invariant cannot be satisfied by being
# uniformly wrong.

# The reference integrator deliberately shares no code with the implementation:
# for a fixed barycentric coordinate the surface restricted to a line is a
# quadratic in one variable, whose sign changes are found in closed form and
# whose integral over each piece is exact. Only the outer integral is discretized.
function reference_area_volume(bezpts::AbstractMatrix{<:Real},
    quantity::String; n::Integer=2000)::Real
    coeffs = bezpts[end,:]
    area = simplex_size(bezpts[1:end-1,[1,3,6]])
    x,w = gausslegendre(n)
    s = (x .+ 1)/2; wt = w/2
    2*area*sum(wt[i]*_line_contribution(coeffs,s[i],quantity) for i = 1:n)
end

function _line_contribution(coeffs, s, quantity)
    L = 1 - s
    L <= 0 && return 0.0
    g(t) = eval_poly([s,t,L-t],coeffs,2,2)
    c = g(0.0); ghalf = g(L/2); gend = g(L)
    a = 2*(gend - 2*ghalf + c)/L^2
    b = (4*ghalf - gend - 3*c)/L
    roots = Float64[]
    if abs(a) < 1e-300
        abs(b) > 0 && push!(roots, -c/b)
    else
        disc = b^2 - 4*a*c
        if disc >= 0
            sq = √disc
            q = -(b + (b < 0 ? -sq : sq))/2
            push!(roots, q/a); push!(roots, c/q)
        end
    end
    edges = unique([0.0; sort(clamp.(filter(isfinite,roots),0.0,L)); L])
    total = 0.0
    for k = 1:length(edges)-1
        lo,hi = edges[k],edges[k+1]
        hi <= lo && continue
        g((lo+hi)/2) < 0 || continue
        total += quantity == "area" ? hi - lo :
            a*(hi^3-lo^3)/3 + b*(hi^2-lo^2)/2 + c*(hi-lo)
    end
    total
end

# A hyperbola crossing the triangle, whose negative region is exactly one third
# of it. The closed form anchors the reference integrator, which in turn anchors
# everything measured against it.
const _sb = sample_simplex(2,2)
const _tri = order_vertices!(reduce(hcat,[[-1.0,0.0],[1.0,0.0],[0.0,1.0]]))
const _pts = barytocart(_sb,_tri)
const _coeffs = getpoly_coeffs(
    [p[1]^2 - (p[2]-1/3)^2 for p = eachcol(_pts)],_sb,2,2)
const _exact_fraction = 1/3

"Map the reference triangle by `M`, keeping the coefficients, and return area/size."
function _fraction_under(M::AbstractMatrix, quantity::String="area")
    pts = M*_pts
    quad_area_volume([pts; _coeffs'],quantity)/simplex_size(pts[:,[1,3,6]])
end

@testset "ScaleInvariance" begin

    @testset "reference integrator" begin
        # Checked before it is trusted anywhere else.
        @test isapprox(reference_area_volume([_pts; _coeffs'],"area"),
            _exact_fraction, rtol=1e-9)
        @test isapprox(quad_area_volume([_pts; _coeffs'],"area"),
            reference_area_volume([_pts; _coeffs'],"area"), rtol=1e-6)
    end

    @testset "coefficient scaling" begin
        # Scaling every coefficient leaves the contour, and so the area, alone,
        # and scales the volume linearly. A fixed tolerance on the coefficients
        # breaks both, which is what made the Fermi surface the worst case: its
        # coefficients are the smallest ones present.
        for scale in [1e10,1e5,1e0,1e-5,1e-20,1e-35,1e-50]
            bez = [_pts; (_coeffs.*scale)']
            @test isapprox(quad_area_volume(bez,"area"),
                _exact_fraction, rtol=1e-12)
            @test isapprox(quad_area_volume(bez,"volume")/scale,
                quad_area_volume([_pts; _coeffs'],"volume"), rtol=1e-12)
        end
    end

    @testset "triangle size" begin
        # Uniform scaling covers the whole Float64 exponent range. This is what a
        # floor on patch size used to truncate: below it the area collapsed to
        # zero while the fraction it should have reported was unchanged.
        for s in [1e0,1e-2,1e-6,1e-12,1e-20,1e-40,1e-80]
            M = [s 0.0; 0.0 s]
            @test isapprox(_fraction_under(M,"area"), _exact_fraction, rtol=1e-12)
            @test isapprox(_fraction_under(M,"volume"),
                _fraction_under([1.0 0.0; 0.0 1.0],"volume"), rtol=1e-12)
        end
    end

    @testset "translation and rotation" begin
        # Neither changes the fraction. Both move the coordinates away from the
        # origin, where the triangle's size is computed from differences of large
        # numbers and loses digits to cancellation.
        for shift in [1e0,1e3], θ in [0.0,0.7,2.5]
            R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            pts = R*_pts .+ shift
            @test isapprox(quad_area_volume([pts; _coeffs'],"area")/
                simplex_size(pts[:,[1,3,6]]), _exact_fraction, rtol=1e-9)
        end
        # By 1e6 the offset costs about ten digits: the unit triangle measures
        # 0.9999999999 and the patch arrives with four intersections. The
        # coordinates carry the same information at any offset, so this is
        # recoverable by working relative to the patch's own centroid.
        for θ in [0.0,0.7,2.5]
            R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            pts = R*_pts .+ 1e6
            @test_broken isapprox(quad_area_volume([pts; _coeffs'],"area")/
                simplex_size(pts[:,[1,3,6]]), _exact_fraction, rtol=1e-9)
        end
    end

    @testset "aspect ratio" begin
        # diag(λ,1/λ) preserves the triangle's size exactly and distorts only its
        # shape, so the fraction must not move. The two directions behave very
        # differently, which is itself the finding: the surface is defined in
        # barycentric coordinates and is unchanged by the map, but parts of the
        # integration reason about the contour in Cartesian coordinates, so the
        # result depends on a distortion the mathematics does not see.
        #
        # Flattening the triangle is tolerated well.
        for λ in [1.0,2.0,5.0,10.0,15.0]
            @test isapprox(_fraction_under([λ 0.0; 0.0 1/λ]), _exact_fraction,
                rtol=1e-10)
        end
        # Stretching it upright is not, and fails from a ratio of 2.
        for λ in [1.0,1.5]
            @test isapprox(_fraction_under([1/λ 0.0; 0.0 λ]), _exact_fraction,
                rtol=1e-10)
        end
    end

    @testset "aspect ratio, known limits" begin
        # Upright stretching gives up at a ratio of 2, and the patch it fails on
        # is the same size at every ratio - 4.7369515e-15 at 2, 3 and 5 alike.
        # A size that does not move with the geometry that produced it is the
        # signature of a tolerance that is still absolute somewhere.
        for λ in [2.0,3.0,5.0]
            @test_broken isapprox(_fraction_under([1/λ 0.0; 0.0 λ]),
                _exact_fraction, rtol=1e-10)
        end
        # Flattening survives further but degrades quietly first: wrong by about
        # a percent at 20, and by less further out, before raising past about 70.
        # Silence is the part worth fixing - an error is recoverable, a plausible
        # wrong Fermi area is not.
        for λ in [20.0,30.0,50.0]
            @test_broken isapprox(_fraction_under([λ 0.0; 0.0 1/λ]),
                _exact_fraction, rtol=1e-10)
        end
        @test_throws ErrorException _fraction_under([100.0 0.0; 0.0 0.01])
        # Ratios of 1e3 and beyond do not raise but fail to terminate in any
        # reasonable time, so they are left out entirely rather than parked here.
    end
end
