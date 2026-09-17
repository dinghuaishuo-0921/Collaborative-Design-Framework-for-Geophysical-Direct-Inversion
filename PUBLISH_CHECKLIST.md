# Publishing checklist for Computers & Geosciences

1. Create a new **public** GitHub repository named
   `Collaborative-Design-Framework-for-Geophysical-Direct-Inversion-` without
   adding a README, `.gitignore`, or license on GitHub; these files are already
   included locally.
2. Open this folder in GitHub Desktop, or add the GitHub repository as the
   `origin` remote from a terminal.
3. Create the first commit and push the `main` branch.
4. Confirm that the repository opens in a private browser window and that
`README.md`, `examples/quick_test.py`, `examples/training_subset_10000.mat`,
and `requirements.txt` can be downloaded without signing in.
5. Run `python examples/quick_test.py` from a clean Python environment, then
   confirm that `results/quick_test_summary.json` is produced.
6. Copy the repository URL from `docs/computer_code_availability.md` into the
   manuscript before the reference list.

The complete source data pools are released separately through Mendeley Data. The
repository includes a 10,000-sample training subset and a synthetic,
format-compatible quick-test example for verification.
