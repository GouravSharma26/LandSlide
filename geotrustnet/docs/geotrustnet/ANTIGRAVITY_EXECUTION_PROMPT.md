# Antigravity Execution Prompt

You are working inside the user's existing LandSlide GitHub repository. Your job is to evolve it into **LandSlide — GeoTrustNet Intelligence**, not to create a separate unrelated project.

## Mandatory reading
Read:
- `/AGENTS.md`
- `/README_GeoTrustNet_Plan.md`
- all files under `docs/geotrustnet/`

## Start with Phase 0 only
1. Inspect repository tree, git branch/status, latest commits, dependencies, configs, tests, data directories, dashboard routes/pages, model code, deployment setup.
2. Run the existing documented tests and dashboard smoke test if feasible. Do not change code to make tests pass during the audit.
3. Verify whether previously described paths/features actually exist.
4. Identify data sources, schemas, licenses, temporal structure, CRS, alignment, label quality, and split strategy.
5. Identify destructive risks and compatibility constraints.
6. Write:
   - `docs/REPOSITORY_AUDIT.md`
   - `docs/IMPLEMENTATION_PLAN.md`
   - `docs/PRESERVATION_AND_ROLLBACK.md`
7. Include a phased file-level implementation plan that maps new GeoTrustNet modules into the existing repository structure.
8. Report exact commands and outcomes, including failures.

## Do not do during Phase 0
- Do not rewrite the dashboard.
- Do not delete existing pages, models, data, or scripts.
- Do not rename the project or change deployment configuration.
- Do not download a large dataset without asking.
- Do not invent benchmark scores or claim research novelty.
- Do not implement the complete model yet.

Stop after Phase 0 and present the audit for user review.

## After approval
Implement one phase at a time according to `docs/geotrustnet/IMPLEMENTATION_ROADMAP.md`. At each phase boundary, write a report under `docs/reports/`, run tests, and stop for approval. Prefer small commits with descriptive messages.
