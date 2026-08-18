# Tolerance robustness

Working tracker for making area, volume and Fermi-level integration correct across
a wide range of coefficients, simplex geometries and level-set types. Source of
truth is this file; it is updated in the same commits that change behaviour.

**Scope.** The code integrates over simplices in both dimensions: triangles in 2D,
tetrahedra in 3D. The interpolated quantity is a quadratic over the simplex, so
the boundary of the region being integrated is a **level curve** on a triangle and
a **level surface** on a tetrahedron. Everything below applies to both unless it
says otherwise. Where a statement is 2D-only that is a gap in the work, not a
limit of the code - and at present most of the measured evidence is 2D, which is
itself one of the larger open items.

## What "robust" has to mean

Band energies are only as good as the Fermi areas underneath them, because the
Fermi level sets the domain the band energy is integrated over. An area that is
quietly wrong by a percent is not a local error. So the target is stated as three
independent axes, each with a range we can name and defend:

1. **Coefficients** — uniform scaling, and coefficients near zero that are still
   meaningful rather than cancellation residue.
2. **Simplex geometry** — size, aspect ratio, orientation, distance from origin,
   for triangles and tetrahedra alike.
3. **Level-set type** — the level curve on a triangle, the level surface on a
   tetrahedron, across every conic or quadric the classifier distinguishes, plus
   the degenerate cases between them: tangency, level sets entirely inside or
   outside the simplex, roots landing exactly on a vertex, edge or face.

## How we decide something is correct

These rules exist because they each caught a real error in this work.

- **Adjudicate against a closed form where one exists, and never against stored
  expectations.** Fermi areas and band energies have closed forms, which are the
  best available answer and should be preferred over any numerical reference. The
  numerical reference integrator is the fallback for cases without one. Stored
  values encode whatever the code did when they were written: an intermediate
  version of the tolerance work moved 18 answers and the tests looked merely
  stale, but measuring showed the old answers were right and the change was a
  regression.
- **Prefer invariants to stored numbers.** Coefficients are barycentric, so
  `area / simplex_size` is invariant under every affine map of the triangle. A
  remaining absolute tolerance breaks that and nothing else.
- **Anchor the reference itself** against closed forms before trusting it. See
  the open item on its accuracy below — it is currently the weakest link on some
  geometries.
- **Show that answers did not move.** Every `quad_area_volume` call the suite
  makes (2257 of them) is captured before and after a change and compared.
- **Add as few tolerances as possible.** Many necessary tolerances are fine; an
  unnecessary one is not. Before adding a constant, check whether an existing
  parameter already means the same thing at the same place - a new name with the
  same value buys nothing and costs a knob. `def_root_boundary_atol` was added
  during this work and then removed: it duplicated the `atol` already threaded
  into `bezcurve_intersects`, where it is the only use of `atol`, so the new
  constant changed no arithmetic and orphaned the caller's control.
- **Assert behaviour, not platform.** Asserting *how* a case failed rather than
  *that* it failed broke Windows CI while five other platforms passed.

## Verified ranges

Measured, and covered by `test/ScaleInvariance.jl` unless noted. **Every row is a
triangle in 2D**; there is currently no equivalent evidence for tetrahedra, which
is tracked as a gap rather than as a pass.

| Axis | Verified (2D) | Notes |
|---|---|---|
| Coefficient scale, uniform | 1e10 → 1e-50 | area invariant, volume linear |
| Simplex size, isotropic | 1e0 → 1e-80 in tests | measured exact to 1e-160 |
| Rotation | all angles | |
| Translation | to 1e3 | |
| Aspect ratio, flattening | to 15 | `diag(λ, 1/λ)` |
| Aspect ratio, upright | to 1.5 | `diag(1/λ, λ)` |
| Level curve: hyperbola | unit scale | in tests |
| Level curve: parallel lines, single line, all-positive, all-negative | unit scale | measured, not yet in tests |
| Level curve: ellipse, parabola, intersecting lines | — | measured but **unadjudicated**: the differences are ~1e-6, which is the reference integrator's own error |
| Any level surface, any tetrahedron | — | not measured |

## Outstanding

Ordered by how badly each one can corrupt a band energy.

- [ ] **Tangent level curve returns half the triangle.** `f = x²` touches zero
      along a line and is never negative, so the area below zero is 0; the code
      returns 0.5. Coefficients are `[1,-1,1,0,0,0]` at unit scale, so this is not
      a scale effect. It is also discontinuous: `x² + 1e-3` gives 0 correctly and
      `x² - 1e-3` gives 0.062 correctly, but exactly at tangency it jumps to 0.5.
      Worst of the open items — silent, large, and reachable at ordinary scales.
- [ ] **Upright stretching fails from aspect ratio 2**, while flattening survives
      to 15. The surface is unchanged by an affine map, so this can only come from
      the parts of the integration that reason about the contour in Cartesian
      coordinates (`getbez_pts_wts`, `conicsection`). The failing patch measures
      4.7369515e-15 at ratios 2, 3 and 5 alike — a size that does not move with
      the geometry producing it is what a still-absolute tolerance looks like.
- [ ] **Flattening degrades silently before it raises**: wrong by about 1% at 20,
      0.2% at 30, 7e-7 at 50, raising past about 70. Non-monotonic, which is
      characteristic of a threshold rather than of conditioning.
- [ ] **Reference integrator is only ~1e-6 on regions with interior turning
      points.** For an ellipse fully inside the triangle the closed form is
      π(0.2)² = 0.1256637061; the implementation matches to nine digits and the
      reference is off by 6.1e-7. Its outer quadrature converges poorly when the
      inner interval length has square-root behaviour. **This blocks adjudicating
      the ellipse, parabola and intersecting-lines cases**, so it should be fixed
      before those are called passing or failing.
- [ ] **Aspect ratios of 1e3 and beyond do not terminate** in reasonable time.
      Excluded from the suite rather than marked broken, since a hanging test is
      worse than an absent one.
- [ ] **Translation to 1e6 costs about ten digits** to cancellation; the unit
      triangle measures 0.9999999999 and the patch arrives with four
      intersections. Likely recoverable by working relative to the patch centroid.
- [ ] **Extreme ratios behave differently per platform**: at 100, Linux, macOS and
      1.12 raise, Windows under 1.10 returns a value. Whatever that value is, it
      is unadjudicated.
- [ ] **Error message dichotomy is incomplete.** A large patch far from the origin
      fails for precision reasons but is reported as "not small, so this indicates
      a genuine problem with the geometry" — which is misleading in exactly the
      case that produces it.

## Harness coverage gaps

What we are not yet testing at all. These are not known failures; they are
unknowns, which is worse.

- [ ] **Level-curve type is not an axis in the suite.** Every assertion in
      `ScaleInvariance.jl` uses one hyperbola. `conicsection` distinguishes seven
      cases and the interesting failures live at the boundaries between them.
- [ ] **Degenerate intersections**: roots exactly on a vertex, roots on an edge,
      curve entering and leaving through the same edge, double roots.
- [ ] **Combined stress**: small triangle *and* anisotropic *and* small
      coefficients together. Each axis is currently tested alone.
- [ ] **3D is half the supported domain and has none of this treatment.**
      Tetrahedra, level surfaces and the volume paths need the same three axes:
      coefficient scaling, simplex geometry including aspect ratio, and quadric
      type. Nothing here should be assumed to carry over from 2D - the splitting
      and the classification are different code.
- [ ] **Closed forms are not yet used as the adjudicator.** Fermi areas and band
      energies have them, which would replace the numerical reference for the
      cases that matter most and remove the accuracy ceiling above. This is
      probably the highest-leverage item in this section: it upgrades the
      instrument every other measurement depends on.
- [ ] **BigFloat**: the noise floor is written to fall with `eps(T)`, and nothing
      exercises it.
- [ ] **End-to-end**: no assertion ties a Fermi area error to a band energy error,
      so we cannot yet say what a given area tolerance buys.

## Tolerance inventory

The preference is to minimise the number of tolerances. Necessary ones are fine;
duplicates are not. Current state worth watching:

- **Five constants share the value 1e-9**: `def_atol`, `def_rtol`,
  `def_coeff_rtol`, `def_simplex_size_rtol`, `def_neighbor_dist_rtol`. They are
  not obviously the same quantity - they are a general absolute tolerance, a
  general relative one, a coefficient scale, a simplex-size fraction and a
  distance fraction - but sharing a number makes it easy to believe a change to
  one is a change to all, and easy to add a sixth that duplicates an existing one.
- `def_atol` has 69 uses across the package and is the workhorse. Any claim that
  the code is free of absolute tolerances has to reckon with that number, since
  each use is a separate question about what "zero" means there.
- [ ] **Audit the remaining `def_atol` uses by dimension**, the way
  `solve_quadratic` and `conicsection` were audited: what does the quantity being
  compared carry the units of, and what degree in the coefficients is it? Uses
  that are genuinely dimensionless keep an absolute tolerance; the rest do not.

## Changes so far

| Commit | Change |
|---|---|
| `9c1ae9b` | Coefficient, extremum and discriminant tests made relative and degree-correct |
| `4370ec3` | Point deduplication scaled to the patch; `def_min_simplex_size` removed |
| `7ca375b` | Unsplittable patches raise instead of being approximated by sign |
| `ab5b0a1` | `test/ScaleInvariance.jl`: affine-invariance harness |
| pending | `def_root_boundary_atol` removed as a duplicate of the `atol` already threaded into `bezcurve_intersects` |
