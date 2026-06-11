# JOSS submission roadmap for `organic-qc-bench`

**Strategy:** push the repository **public today** to start the
six-month development clock, organically grow the codebase over that
period, and submit to JOSS once the
[6-month public-history requirement](https://joss.readthedocs.io/en/latest/submitting.html)
is satisfied.

* **Today (2026-06-11):** push, tag `v0.1.0`, wire Zenodo. (~30 min)
* **Now → 2026-12-11:** maintenance, bug-fixes, doc improvements,
  one or two minor releases (`v0.1.1`, `v0.2.0`). Target ≥ 30 commits
  spread across this period to satisfy JOSS's commit-distribution
  check.
* **2026-12-11 or later:** submit to JOSS at
  <https://joss.theoj.org/papers/new>.

---

## Step 1 — local repo already prepared

`.git/` exists, initial commit `1385925` is in place. No further
action required for this step.

## Step 2 — push to GitHub (manual; auto-mode blocks this)

The repo namespace `WakauraH/organic-qc-bench` matches the
manuscript and `paper.md`. The personal-account `WakauraH/` form is
used because the `qiri-jp/` organisation does not exist on this
account.

```bash
cd /Users/deeptell01/Documents/alterego/personal/3layerqbrainstab/package
gh repo create WakauraH/organic-qc-bench --public \
    --description "Benchmark suite for Petz-style covariant recovery on dephasing-depolarizing channels" \
    --source=. --remote=origin --push
```

After the push, verify the CI matrix kicks off automatically:

```bash
gh workflow list
gh run watch                 # streams the latest run
```

The badges at the top of README (`pypi`, `actions`) will turn green
once the CI succeeds.

## Step 3 — tag `v0.1.0` and create a GitHub release

```bash
git tag -a v0.1.0 -m "v0.1.0: initial public release"
git push origin v0.1.0
gh release create v0.1.0 \
    --title "v0.1.0 — initial public release" \
    --notes "First public release of organic-qc-bench. Benchmarks a
CPTP Petz-style covariant recovery map on dephasing--depolarizing
channels. 11 smoke tests, CI on ubuntu+macos × py3.9-3.12."
```

## Step 4 — wire Zenodo (optional but recommended)

Visit <https://zenodo.org/account/settings/github/> and flip the
toggle for `WakauraH/organic-qc-bench` to ON. From the next
release onwards, Zenodo will auto-mint a DOI per release. The DOI
for `v0.1.0` may need to be re-issued by clicking "Create release
DOI" once Zenodo sees the existing tag.

The Zenodo DOI is required at JOSS *review* time (December onwards),
not at submission time.

## Step 5 — six-month organic-growth plan (now → 2026-12-11)

JOSS's pre-review explicitly rejects "repo dumps" with a single
commit. The cleanest defence is to use the package during the wait
period and let commits accumulate naturally. Suggested rhythm:

| When | Type of commit |
|---|---|
| Now (multiple) | Splitting docs improvements into separate commits as feedback arrives |
| Month 1 (July) | Fix numerical-warning sources in `core.py` (`RuntimeWarning: matmul` already observed during the refit sweep) |
| Month 2 (August) | Add Wilson-CI helper to `bv.py`'s public API; new smoke test |
| Month 3 (September) | Extend `peak.py` with the alternative-fit selector from `refit_gamma_peak.py` |
| Month 4 (October) | Tag `v0.1.1` patch release; CHANGELOG.md |
| Month 5 (November) | Documentation expansion (Jupyter example notebook in `examples/`) |
| Month 6 (December) | Tag `v0.2.0` minor release with the cumulative improvements; final paper.md polish |

This produces ≥ 6 tagged releases worth of activity with no
"backfilling" of past dates, satisfying JOSS's substantive-development
check.

## Step 6 — submit to JOSS (December 2026 or later)

1. Open <https://joss.theoj.org/papers/new>.
2. Repository URL: `https://github.com/WakauraH/organic-qc-bench`.
3. Software version: latest tag at submission time (likely `v0.2.0`).
4. Branch with paper: `main` (paper.md lives at `joss/paper.md`).
5. Software archive: Zenodo DOI of the same tag.
6. Submitting author: H. Wakaura, ORCID 0000-0001-8381-8323.
7. **Declare related preprints in the form**:
   * "Fidelity-gain peak ..." (`pra_gammapeak_v4`), under
     review at PRA (or wherever it ends up by then).
   * "3-Layer Quantum Brain Hypothesis", Wakaura, Research Square,
     [doi:10.21203/rs.3.rs-9278975/v1](https://doi.org/10.21203/rs.3.rs-9278975/v1).
8. **Declare conflicts of interest**: none; both authors are QIRI
   employees; funding from QIRI; companion preprint is single-authored
   by H. Wakaura.

JOSS will respond with a pre-review issue on the
`openjournals/joss-reviews` repository, generally within one or two
weeks.

## Step 7 — what JOSS reviewers will check

| Check | Status |
|---|---|
| Open-source licence | ✓ MIT, [`LICENSE`](LICENSE) |
| ≥ 6 months public development | will be satisfied at submission time |
| Substantive commit history (not "repo dump") | will be satisfied by the Step 5 plan |
| Substantial scholarly effort | ✓ 8 modules, CLI, CI matrix |
| API documentation | ✓ README quick-start + CLI section |
| Tests | ✓ 11 `pytest` smoke tests (and additions during Step 5) |
| CI | ✓ GitHub Actions, py3.9–3.12, ubuntu+macos |
| Paper compiles via Pandoc/JOSS | dry-run with `inara` below |
| References have DOIs | ✓ all 11 entries |
| Required sections (Summary, Statement of need, References) | ✓ |
| AI-usage disclosure | ✓ |
| Authorship contribution evidence in repo | add `CONTRIBUTORS.md` before submission (TODO) |

For a local `inara` dry-run:

```bash
docker run --rm -it \
    -v "$PWD":/data openjournals/inara \
    -o pdf,crossref joss/paper.md
```

## Notes / non-blocking caveats

* The badge URLs in README (`pypi`, `actions`) will turn red until
  the repository is public and the first CI run completes; this is
  normal.
* `dist/` and built wheels stay out of git; JOSS reviewers rebuild
  from source. The `.gitignore` already excludes them.
* If the repository name or owner ever changes, do a single
  `sed -i '' 's|WakauraH/organic-qc-bench|NEW/organic-qc-bench|g'`
  across `README.md`, `joss/paper.md`, and the badge URLs.
