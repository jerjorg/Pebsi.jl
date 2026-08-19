# Tolerance robustness

Tracker for making area, volume and Fermi-level integration correct across a wide
range of coefficients, simplex geometries and level-set types. This file is the
source of truth and is updated in the same commits that change behaviour.

## Where this stands

**2D is in good shape and measured. 3D runs for the first time and does not
converge.**

The tolerance work did what it set out to do in two dimensions: area and volume
are now exact across the whole Float64 exponent range of simplex sizes, across
sixty orders of coefficient scale, and under rotation, translation and moderate
distortion. Every one of those is a property test rather than a stored number.

The work then spent a long time on the far tail of aspect ratios, which was the
wrong place to be. The end-to-end check that should have come first - integrate a
3D EPM and compare against the true value stored in the code - showed that no 3D
calculation could run at all, and that once it could, it does not converge. That
is a much larger gap than anything remaining on the tolerance axis, and it is
where the next work belongs.

## Scope

The code integrates over simplices in both dimensions: triangles in 2D,
tetrahedra in 3D. The interpolated quantity is a quadratic over the simplex, so
the boundary of the region being integrated is a **level curve** on a triangle and
a **level surface** on a tetrahedron. Everything below applies to both unless it
says otherwise; almost all the measured evidence is 2D, which is itself a gap.

## What "robust" has to mean

Band energies are only as good as the Fermi areas underneath them: the Fermi level
sets the domain the band energy is integrated over, so an area that is quietly
wrong is not a local error. Three axes, each with a range that can be named and
defended:

1. **Coefficients** — uniform scaling, and coefficients near zero that are
   meaningful rather than cancellation residue.
2. **Simplex geometry** — size, aspect ratio, orientation, distance from origin,
   magnitude of the vertex components, for triangles and tetrahedra alike.
3. **Level-set type** — every conic or quadric the classifier distinguishes, plus
   the degenerate cases between them: tangency, level sets wholly inside or
   outside, roots on a vertex, edge or face.

## What was implemented

| Commit | Change | Evidence |
|---|---|---|
| `9c1ae9b` | Coefficient, extremum and discriminant tests made relative and degree-correct; the three conic invariants given tolerances of matching degree | coefficient scaling exact 1e10 → 1e-50; 2257/2257 calls unchanged |
| `4370ec3` | Point deduplication scaled to the patch; `def_min_simplex_size` removed entirely | simplex size exact to 1e-160, was failing below 1e-12 |
| `c315b4f` | `test/ScaleInvariance.jl`: affine-invariance harness plus an independent reference integrator | — |
| `45bbfee` | `def_root_boundary_atol` removed as a duplicate of an existing `atol` | no arithmetic changed |
| `2c72a6c` | Vertex component magnitude added as its own axis | exact 1e-8 → 1e8 |
| `c61a909` | One-signed patches decided by the surface's exact range over the simplex, before the intersection logic | tangency bug: `x²` returned half the triangle, now 0; 2257/2257 unchanged |
| `c6e39b2` | The on-edge test scaled to the triangle instead of a fixed 1e-9 | upright aspect ratio 2 → 3; of 16 changed calls, 14 closer to the reference, 0 further |
| `865bb69` | Crossing counts carried alongside patches while splitting, instead of recounting all of them after every split | 3× faster where splitting is heavy; bit-identical |
| `9e2ae80` | `ibz_borders` fixed in 3D — it indexed the hull's points with a facet's coordinates | 3D ran end to end for the first time |
| `d9167ac` | Unsplittable patches integrated by sign again, but with a warning | 3D no longer aborts; upright ratios return to 15 at ~1e-11 |

Reversed along the way, and worth knowing about: `7ca375b` made unsplittable
patches a hard error, justified by the branch being unreachable. It was reachable
in 3D, which could not run when that was measured. `d9167ac` undoes it.

## Verified ranges

Measured, and covered by `test/ScaleInvariance.jl` unless noted. **Every row is 2D.**

| Axis | Verified | Notes |
|---|---|---|
| Coefficient scale, uniform | 1e10 → 1e-50, exact | area invariant, volume linear |
| Simplex size, isotropic | 1e0 → 1e-160, exact | tested to 1e-80 |
| Rotation | all angles, exact | |
| Translation | to 1e3, exact | 1e6 loses ~10 digits to cancellation |
| Vertex component magnitude | 1e-8 → 1e8, exact | independent of size and translation |
| Aspect ratio, flattening | to 15, exact | 20 → 30 wrong by 1.2% → 0.2%; 50 → 100 wrong at ~1e-6 |
| Aspect ratio, upright | to 1.5 exact; to 15 at ~1e-11 | beyond 1.5 the answer comes through sign-integrated patches and carries a warning |
| Level curve: hyperbola, tangent, one-signed, circle wholly inside | unit scale | circle anchored on πr² |
| Level curve: parallel lines, single line | unit scale | measured, not in tests |
| Level curve: ellipse, parabola, intersecting lines | **unadjudicated** | differences ~1e-6, which is the reference integrator's own error |
| 3D, anything | **does not converge** | see below |
| Any level surface, any tetrahedron | not measured | |

## What is left

Ordered by what most affects a band energy.

- [ ] **3D does not converge.** For Ag, over 24 refinements the band energy error
      falls from 8.0e-2 to 1.0e-2 and then wanders between 5e-3 and 1.2e-2. The
      stored true value carries an uncertainty of 5.1e-6, so the calculation stops
      about three orders of magnitude short. The Fermi level behaves the same way:
      6.1e-5 at one step, 3.7e-3 six steps later.
- [ ] **Refinement barely refines in 3D.** Each step splits one simplex and adds
      six k-points, taking the mesh from 112 to 262 over 24 refinements. An error
      that rises as often as it falls, on a mesh growing that slowly, points at
      the refinement choosing the wrong simplex rather than at the integration
      being imprecise. **This is the place to start.** It decides whether anything
      else on this list matters.
- [ ] **Subdivision cannot break up 3- and 4-crossing patches.** At an upright
      ratio of 5, splitting makes 320 pieces and leaves 10 that `split_bezsurf₁`
      declines to subdivide at all: every candidate sub-triangle comes back
      touching the padding box. They are now integrated by sign with a warning
      rather than aborting, which is a usable answer and not a correct one.
      The crossings are genuine - 24% to 100% of the patch's extent apart, on
      patches that are nearly equilateral rather than slivers, classified
      `hyperbola` with every crossing a real sign change.
- [ ] **The reference integrator is only ~1e-6 where the region has interior
      turning points.** For a circle wholly inside the triangle the closed form is
      π(0.2)² = 0.1256637061; the implementation matches to nine digits and the
      reference is off by 6.1e-7. Its outer quadrature converges poorly through the
      square-root kink where the region's extent vanishes. This blocks adjudicating
      the ellipse, parabola and intersecting-lines cases, and it is the instrument
      every other measurement leans on. Splitting the outer integral at the turning
      points is the direct fix.
- [ ] **Flattening is silently wrong between 20 and 50** — 1.2% at 20, 0.2% at 30,
      4e-7 at 50 — and returns ~1e-6 errors at 70 and 100. Non-monotonic, which is
      characteristic of a threshold rather than of conditioning.
- [ ] **Results near the limit depend on the scipy version.** Subdivision
      triangulates through scipy's Delaunay, and versions split near-degenerate
      slivers differently, so the same aspect ratio returns an answer on one
      machine and raises on another. For a package whose point is a reproducible
      Fermi area this is worth more than a footnote. A Julia-native triangulation
      would remove it, and would also lift the Float64-only restriction PyCall
      imposes on everything downstream — which is the same thing blocking BigFloat.
- [ ] **Translation to 1e6** costs about ten digits to cancellation; the unit
      triangle measures 0.9999999999. Recoverable by working relative to the
      patch's own centroid.
- [ ] **Extreme ratios do not terminate.** At 1e3 and beyond the subdivision runs
      without end; around vertex components of 1e8 it raises `StackOverflowError`,
      which Julia reports as possible state corruption. Subdivision wants a depth
      or progress bound on its own account.
- [ ] **Audit the remaining `def_atol` uses by dimension**, the way
      `solve_quadratic` and `conicsection` were audited: what units does the
      compared quantity carry, and what degree in the coefficients? There are 69
      of them, and any claim this code is free of absolute tolerances has to
      reckon with that number. Five constants share the value 1e-9 - `def_atol`,
      `def_rtol`, `def_coeff_rtol`, `def_simplex_size_rtol`,
      `def_neighbor_dist_rtol` - which makes it easy to add a sixth duplicate.
- [ ] **The padding box in `split_bezsurf₁` is asymmetric, and load-bearing.**
      `xmin` is derived from an already-enlarged `xmax`, so one side is padded by
      100 widths and the other by 10100. Plainly not the intent, but making it
      symmetric regresses the upright ratio of 3 from working to raising. Worth
      understanding before it is tidied.
- [ ] **3D has none of the treatment 2D has had.** Tetrahedra and level surfaces
      need the same three axes. Enumerate the surface types - ellipsoid, both
      hyperboloids, both paraboloids, cone, the three cylinders, intersecting,
      parallel and single planes, point, empty - and the configurations, which
      matter as much: cutting off one vertex, cutting off two, entering and leaving
      through one face, splitting the tetrahedron so neither piece holds a vertex,
      tangent to a face, an edge or a vertex, wholly inside, wholly outside.
      Walking exactly this list one dimension down is what found the tangency bug.
- [ ] **BigFloat is untested.** The noise floor is written to fall with `eps(T)`,
      and nothing exercises it. Blocked in practice by the scipy triangulation
      being Float64-only.

## How we decide something is correct

Each of these was earned by getting it wrong first.

- **Adjudicate against something independent of the code, never against stored
  expectations.** Fermi areas and band energies have closed forms, but the closed
  forms *in the code* are the two-intersection case's analytic solution, so
  checking the code against them is circular. What is independent: the numerical
  reference integrator, and closed forms for hand-constructed cases whose geometry
  is known outside the code. An intermediate version of this work moved 18 answers
  and the tests looked merely stale; measuring showed the old answers were right
  and the change was a regression.
- **Prefer invariants to stored numbers.** Coefficients are barycentric, so
  `area / simplex_size` is invariant under every affine map of the simplex. A
  remaining absolute tolerance breaks that and nothing else.
- **Check that an invariant is one.** A parity check on the crossing count was
  proposed here, on the grounds that a closed curve meets a boundary an even
  number of times. The level set of a quadratic is not always closed - a
  hyperbola's branches are unbounded - so odd counts are ordinary geometry, and
  the check would have thrown away good patches.
- **Show the answers did not move.** Every `quad_area_volume` call the suite makes
  (2257) is captured before and after a change and compared. Where they do move,
  adjudicate each one.
- **Add as few tolerances as possible.** Many necessary tolerances are fine; an
  unnecessary one is not. Check whether an existing parameter already means the
  same thing at the same place before adding a constant.
- **Assert behaviour, not platform.** Asserting *how* a case failed rather than
  *that* it failed broke Windows CI while five other platforms passed. Where a
  result is platform-dependent, assert that an answer, if one comes back, is
  right.
- **Measure before attributing.** Several confident explanations here were wrong -
  that the noise floor caused a regression, that recounting made the degenerate
  conics slow, that a fixed patch size implied a stale tolerance. Each took one
  measurement to disprove.

## Tools

- `test/ScaleInvariance.jl` — the affine-invariance harness, and
  `reference_area_volume`, an integrator sharing no code with the implementation:
  for a fixed barycentric coordinate the surface restricted to a line is a
  quadratic whose sign changes and integral are both closed form, so only the
  outer integral is discretized.
- `test/ThreeDimensional.jl` — a loose guard that 3D runs at all, which is what
  was missing when it silently did not.
- **Corpus capture** — instrument `quad_area_volume` to push
  `(bezpts, quantity, result)` into a global, run the suite, serialize. Comparing
  two such captures is what made every change above verifiable.
