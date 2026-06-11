# JOSS submission steps for `organic-qc-bench`

## Step 1 – initialise local git repo (one-time)

```bash
cd /Users/deeptell01/Documents/alterego/personal/3layerqbrainstab/package
git init -b main
git add LICENSE README.md pyproject.toml \
        .gitignore .github/workflows/test.yml \
        src/organic_qc_bench/ tests/ joss/ JOSS_SUBMIT.md
git commit -m "Initial release: organic-qc-bench v0.1.0"
```

## Step 2 – create the GitHub repository and push

`gh` is already authenticated as **WakauraH**. The expected repo
slug used throughout the manuscript and the JOSS paper is
`WakauraH/organic-qc-bench`. If the `qiri-jp` organisation does not
exist on your account, replace it with `WakauraH` (your personal
account) and update README + paper.md accordingly.

```bash
# If qiri-jp org exists:
gh repo create WakauraH/organic-qc-bench --public \
    --description "Benchmark suite for Petz-style covariant recovery on dephasing-depolarizing channels" \
    --source=. --remote=origin --push

# Otherwise (personal namespace):
gh repo create WakauraH/organic-qc-bench --public \
    --description "Benchmark suite for Petz-style covariant recovery on dephasing-depolarizing channels" \
    --source=. --remote=origin --push
```

After the push, verify the CI matrix kicks off automatically:

```bash
gh workflow list
gh run watch                 # streams the latest run
```

The badge at the top of README expects the workflow named `tests`
to exist and to be passing.

## Step 3 – tag the release

Required by JOSS: the submission must reference a tagged release that
matches the version metadata in the paper.

```bash
git tag -a v0.1.0 -m "v0.1.0: JOSS submission"
git push origin v0.1.0
gh release create v0.1.0 \
    --title "v0.1.0 — JOSS submission" \
    --notes "Initial public release for JOSS review."
```

This produces a Zenodo-archived DOI if Zenodo↔GitHub is wired (set
up under <https://zenodo.org/account/settings/github/>). The DOI is
required at JOSS review time but not at submission time.

## Step 4 – submit to JOSS

1. Open <https://joss.theoj.org/papers/new>.
2. Repository URL: `https://github.com/WakauraH/organic-qc-bench`
   (or `WakauraH/organic-qc-bench`).
3. Software version: `v0.1.0` (matches the git tag).
4. Branch with paper: `main` (paper.md lives at `joss/paper.md`).
5. Software archive: leave blank until pre-review opens; you will
   then attach the Zenodo DOI.
6. Submitting author: H. Wakaura, ORCID 0000-0001-8381-8323.

JOSS will reply within a day or two with a pre-review issue on
their `joss-reviews` repository.

## Step 5 – what the JOSS reviewers will check

| Check | Status |
|---|---|
| Open-source licence | ✓ MIT, [`LICENSE`](LICENSE) |
| Software performs a non-trivial task | ✓ recovery-map benchmark |
| API documentation | ✓ README quick-start + CLI section |
| Tests | ✓ 11 `pytest` smoke tests |
| CI | ✓ GitHub Actions, py3.9–3.12, ubuntu+macos |
| Paper ≤ 1000 words | ✓ 898 words |
| Paper compiles via Pandoc/JOSS | run `pandoc joss/paper.md -o paper.pdf` locally to spot-check |
| References have DOIs | ✓ all 11 entries |
| Statement of need | ✓ explicit subsection |

If you would like a dry-run, install `inara` (JOSS's reference
compiler) and run:

```bash
pip install pyjoss-inara          # if you want a local Pandoc fallback
# or use the official JOSS docker image:
docker run --rm -it \
    -v "$PWD":/data openjournals/inara \
    -o pdf,crossref joss/paper.md
```

## Notes / non-blocking caveats

* The badge URLs in README (`pypi`, `actions`) will turn red until
  the repository is public and the first CI run completes; this is
  normal.
* The paper's repository URL is set to `WakauraH/organic-qc-bench`;
  if the eventual namespace differs, do a single
  `sed -i '' 's|WakauraH/organic-qc-bench|YOUR_NAMESPACE/organic-qc-bench|g'`
  across README.md, joss/paper.md and the badge URLs before pushing.
* Do **not** include `dist/` or the locally-built wheels in the
  initial commit; JOSS reviewers will rebuild from source.
