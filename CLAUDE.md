# AGENTS for `uubed-docs` (Documentation)

This repository is dedicated to the documentation and educational materials for the `uubed` project.

## Role of this Repository:
- **User Documentation:** Houses quickstart guides, API references, and usage examples.
- **Technical Book:** Contains the content for "The QuadB64 Codex," explaining the underlying theory and implementation details.
- **Documentation Infrastructure:** Manages the build and deployment of the documentation website.

## Key Agents and Their Focus:
- **Documentation Specialist:** Ensures clarity, accuracy, and completeness of all user-facing documentation.
- **Technical Writer:** Develops the content for "The QuadB64 Codex," translating complex technical concepts into accessible language.
- **UX Designer:** Focuses on the usability and navigability of the documentation website.

If you work with Python, use 'uv pip' instead of 'pip', and use 'uvx hatch test' instead of 'python -m pytest'. 

When I say /report, you must: Read all `./TODO.md` and `./PLAN.md` files and analyze recent changes. Document all changes in `./CHANGELOG.md`. From `./TODO.md` and `./PLAN.md` remove things that are done. Make sure that `./PLAN.md` contains a detailed, clear plan that discusses specifics, while `./TODO.md` is its flat simplified itemized `- [ ]`-prefixed representation. When I say /work, you must work in iterations like so: Read all `./TODO.md` and `./PLAN.md` files and reflect. Work on the tasks. Think, contemplate, research, reflect, refine, revise. Be careful, curious, vigilant, energetic. Verify your changes. Think aloud. Consult, research, reflect. Then update `./PLAN.md` and `./TODO.md` with tasks that will lead to improving the work you’ve just done. Then '/report', and then iterate again.