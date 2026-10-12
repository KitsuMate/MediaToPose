# Releasing

## Before tagging

1. Set `version` in `media_to_pose/blender_manifest.toml`; it is the only
   place the version is written.
2. Add the release to `CHANGELOG.md`.
3. If `README.md`, `LICENSE` or `THIRD_PARTY_NOTICES.md` changed, copy them
   into `media_to_pose/`; `tests/unit/test_release_files.py` checks the
   copies.
4. Run the checks in `AGENTS.md` (Ruff, pyright, the unit tier and the
   Blender suite).
5. Build and check the packages locally:

   ```
   blender --command extension build --source-dir media_to_pose \
       --output-dir dist --split-platforms
   python ci/validate_distributions.py --dist dist
   python ci/run_release_smoke.py --blender <blender> \
       --archive dist/media_to_pose-<version>-<platform>.zip
   ```

   The release smoke installs the archive into an empty profile, exercises
   registration, the bundled dependencies and the `media_to_pose` command
   line, then removes and reinstalls it.

## Publishing

Pushing a version tag builds and validates the platform packages, then creates
the GitHub release without manual approval.

Publishing the same packages to Blender Extensions uses the protected GitHub
environment named `blender-extensions`. Configure this environment in the
repository settings before the first release:

1. Add at least one required reviewer.
2. Add `BLENDER_EXTENSIONS_TOKEN` as an environment secret.
3. Do not add the GitHub release job to this environment.

After the GitHub release is available, GitHub Actions waits for environment
approval before starting the Blender Extensions publishing job.
