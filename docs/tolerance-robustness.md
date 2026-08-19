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

- **Adjudicate against something independent of the code, and never against
  stored expectations.** Fermi areas and band energies do have closed forms, but
  the closed forms *in the code* are the analytic solution of the two-intersection
  case - so checking the code against them is circular and proves nothing. Two
  things are genuinely independent: the numerical reference integrator, and a
  closed form for a hand-constructed case whose geometry is known outside the code
  (an ellipse's area is πr² whatever the implementation does). Use those. Stored
  values encode whatever the code did when they were written: an intermediate
  version of the tolerance work moved 18 answers and the tests looked merely
  stale, but measuring showed the old answers were right and the change was a
  regression.
- **Prefer invariants to stored numbers.** Coefficients are barycentric, so
  `area / simplex_size` is invariant under every affine map of the triangle. A
  remaining absolute tolerance breaks that and nothing else.
- **Check an invariant is one before relying on it.** A parity check on the
  crossing count was proposed here on the grounds that a closed curve meets a
  boundary an even number of times. The level set of a quadratic is not always a
  closed curve - a hyperbola's branches are unbounded - so odd counts are ordinary
  geometry, and the check would have discarded good patches.
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
| Vertex component magnitude | 1e-8 → 1e8 | exact at every magnitude, holding shape in range; isolates cleanly from size and translation |
| Aspect ratio, flattening | to 15 | `diag(λ, 1/λ)` |
| Aspect ratio, upright | exact to 1.5; to 15 at ~1e-11 | beyond 1.5 the answer comes through sign-integrated patches, and whether it comes at all depends on the scipy version |
| Level curve: hyperbola | unit scale | in tests |
| Level curve: tangent, one-signed, circle wholly inside | unit scale | in tests; circle anchored on pi*r^2 |
| Level curve: parallel lines, single line, all-positive, all-negative | unit scale | measured, not yet in tests |
| Level curve: ellipse, parabola, intersecting lines | — | measured but **unadjudicated**: the differences are ~1e-6, which is the reference integrator's own error |
| Any level surface, any tetrahedron | — | not measured |
| 3D end to end (Ag) | **does not converge** | runs now, but the band energy plateaus ~1e-2 from the stored value, which carries an uncertainty of 5.1e-6 |

## Outstanding

Ordered by how badly each one can corrupt a band energy.

- [ ] **3D does not converge to the stored band energy.** With `ibz_borders`
      fixed, Ag runs end to end for the first time. Over 24 refinements the band
      energy error falls from 8.0e-2 to 1.0e-2 and then stops improving, moving up
      and down between 5e-3 and 1.2e-2; the stored value carries an uncertainty of
      5.1e-6, so the calculation is about three orders of magnitude short of its
      own target. The Fermi level behaves the same way - it touches 6.1e-5 at one
      step and is back at 3.7e-3 six steps later.
- [ ] **Refinement barely refines in 3D.** Each step splits one simplex and adds
      six k-points: 112 to 262 over 24 refinements. Whatever the integration is
      doing, the mesh is not being driven anywhere near hard enough to tell, and
      the non-monotonic error is consistent with a refinement criterion choosing
      the wrong simplex rather than with an integration that is merely imprecise.
      This wants looking at before any more tolerance work, because it is the
      thing that decides whether the rest of it matters.

- [ ] **Upright stretching is approximate beyond 1.5**, and flattening is wrong
      between 20 and 50. Upright now returns answers out to 15, but from 2 onward
      they come through patches too small to subdivide that are integrated by
      their sign: right to about 1e-11 rather than exactly, and accompanied by a
      warning. That is a usable answer, not a correct one, and the underlying
      subdivision failure is unchanged. Was 2, until the test for the curve's midpoint lying on an edge was
      scaled to the triangle: it compared a distance in coordinate space against a
      fixed 1e-9, which on a patch of extent 1e-7 is a hundredth of the whole
      patch, so points well inside were called "on the edge" and their
      intersections discarded. Two orientations of the same barycentric surface
      told the story - flattening never reached that test at all, while upright
      reached it 46 times and it fired on 22.
      What remains is the same shape of problem one level down: subdivision
      produces slivers whose intersection count it never reduces, and recurses
      until they reach the relative precision floor at about 21 eps of the parent.
      What is actually stuck: at a ratio of 5 the splitting produces 320 pieces and
      leaves 10 of them with 3 or 4 crossings that `split_bezsurf1` will not
      subdivide at all - every candidate sub-triangle comes back touching the
      padding box, so it returns its input unchanged. The crossings are genuinely
      distinct, 24% to 100% of the patch's extent apart, and one such patch is
      nearly equilateral rather than a sliver, so this is not roots being
      double-counted or a degenerate shape. Four crossings is a real
      configuration - the level set entering and leaving through one edge - that
      the two-intersection formula cannot take and the splitting has to break up.
      A count of 3 is not a miscount: the level set of a quadratic is not
      always a closed curve. A hyperbola has two unbounded branches, so the region
      below it can meet the triangle's boundary an odd number of times - one
      branch crossing while the other touches, or clips a corner. The patches that
      stick here are classified `hyperbola`, and their three crossings are all
      genuine sign changes. So the work is to integrate 3- and 4-crossing
      configurations, or to subdivide them successfully, rather than to explain
      the count away.
- [ ] **The padding box in `split_bezsurf1` is asymmetric, and load-bearing.**
      `xmin` is derived from the already-enlarged `xmax`, so the patch is padded by
      100 widths on one side and 10100 on the other. That is plainly not the
      intent, but making it symmetric is a regression: the upright ratio of 3 goes
      from working to raising. Something about the lopsided box is helping the
      triangulation, and it should be understood before it is corrected.
- [ ] **Deep enough recursion overflows the stack.** At vertex components around
      1e8 the upright case raises `StackOverflowError`, and Julia reports that
      program state may be corrupted. This is the same non-termination as the
      large-aspect-ratio hang, far enough along to be a memory-safety concern
      rather than a slow path, so subdivision needs a depth or progress bound
      regardless of what fixes the underlying geometry problem.
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
- [ ] **Results near the limit depend on the scipy version.** The subdivision
      triangulates through scipy's Delaunay, and different versions split these
      near-degenerate slivers differently, so the same aspect ratio returns an
      answer on one machine and raises on another - upright 2 and 3 do exactly
      that. At a flattening ratio of 100, Linux, macOS and 1.12 raise while
      Windows under 1.10 returns a value. For a package whose point is a
      reproducible Fermi area this is worth more than a footnote: the boundary of
      what works is not a property of the code alone. Replacing the triangulation
      with a Julia-native one would remove the dependence, and would also lift the
      Float64-only restriction that PyCall imposes.
- [ ] **Error message dichotomy is incomplete.** A large patch far from the origin
      fails for precision reasons but is reported as "not small, so this indicates
      a genuine problem with the geometry" — which is misleading in exactly the
      case that produces it.

## Harness coverage gaps

What we are not yet testing at all. These are not known failures; they are
unknowns, which is worse.

- [ ] **Level-curve type is only partly an axis in the suite.** Tangency,
      one-signed surfaces and a circle wholly inside are covered; the ellipse
      crossing an edge, the parabola and the intersecting-lines cases are still
      unadjudicated, waiting on the reference integrator's accuracy. The
      degenerate-conic cases are also slow enough at unit scale - tens of seconds
      - to be worth a look on their own account.
- [ ] **Degenerate intersections**: roots exactly on a vertex, roots on an edge,
      curve entering and leaving through the same edge, double roots.
- [ ] **Combined stress**: small triangle *and* anisotropic *and* small
      coefficients together. Each axis is currently tested alone.
- [ ] **3D is half the supported domain and has none of this treatment.**
      Tetrahedra, level surfaces and the volume paths need the same three axes:
      coefficient scaling, simplex geometry including aspect ratio, and quadric
      type. Nothing here should be assumed to carry over from 2D - the splitting
      and the classification are different code.
- [ ] **Enumerate the level surfaces and their intersections with a tetrahedron**,
      as the conic types were enumerated in 2D. The surface types: ellipsoid,
      hyperboloid of one and of two sheets, elliptic and hyperbolic paraboloid,
      cone, elliptic/hyperbolic/parabolic cylinder, intersecting planes, parallel
      planes, single plane, point, empty. The configurations matter as much as the
      types: cutting off one vertex, cutting off two, entering and leaving through
      the same face, cutting the tetrahedron into two pieces neither of which
      contains a vertex, tangent to a face, an edge or a vertex, and lying wholly
      inside or wholly outside. The 2D failure that matters most - tangency - was
      found by walking exactly this list one dimension down.
- [ ] **Raise the numerical reference's accuracy.** It is the instrument every
      other measurement depends on, and its ~1e-6 ceiling on some geometries is
      the binding constraint on what can be adjudicated. The closed forms are not
      a way around this: the ones in the code are the two-intersection case's
      analytic solution, so measuring the code against them is circular. Fixing
      the outer quadrature - splitting the integration at the turning points
      rather than pushing Gauss-Legendre through a square-root kink - is the
      direct route.
- [ ] **Build a library of hand-constructed cases with known geometry**, where the
      answer follows from the construction rather than from any implementation: a
      circle of known radius entirely inside, a half-plane cutting a known
      fraction, a region congruent to one whose area is elementary. These are
      independent of the code in a way the code's own closed forms are not, and
      they are what anchors the reference integrator itself.
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
| `45bbfee` | `def_root_boundary_atol` removed as a duplicate of the `atol` already threaded into `bezcurve_intersects` |
| `2c72a6c` | Vertex component magnitude measured and covered; two tracker claims corrected |
| `c61a909` | Tangency fixed: the range over the simplex is computed exactly and settles the one-signed cases before the intersection logic runs |
| pending | The on-edge test scaled to the triangle; upright aspect ratio extends from 2 to 3, and 16 captured calls improve |
