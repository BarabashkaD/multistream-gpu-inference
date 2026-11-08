# Container distribution evidence

This directory is not legal approval. Public registry upload, image archive delivery and stable releases remain disabled until the actual distribution has been reviewed.

Each built image has `/app/compliance/inventory.json`: installed Python versions, original package license/notice locations and hashes, Debian package versions, and bundled FFmpeg source evidence. It is a partial evidence inventory, not a complete SBOM: static/nested libraries may require manual inspection. Installed wheel contents under `/opt/venv` and Debian copyright files under `/usr/share/doc` remain in place.

## Verified evidence and unresolved review

- PyNvVideoCodec 2.2.3 CP314 Linux x86_64 wheel SHA-256: `ffa03cedc4128456727fd5d1cf5a91fa747cda6cf6544d1b603b1b9d84d00eae` (PyPI). It supplies MIT/LICENSE, LGPL/NOTICES, FFmpeg 9.0.1 sources and a readme. The readme says libavcodec is statically embedded in libavformat. The build checks that its source archive survives installation and final-stage copying. Correspondence to shipped binaries, patches, build information and replacement/relinking obligations are still to be verified. Supplied sources alone are not proof of compliance. Before release verify access to both GPLv3 and LGPLv3 texts and the applicable library replacement/relinking requirements.
- YOLO26n: exporter pins `ultralytics/assets` release `v8.4.0`, weight hash from GitHub release metadata, then verifies downloaded bytes before loading. Export manifests record package versions, parameters and output hashes. This records provenance, not a new license grant. Model licensing and the Ultralytics-derived tracker remain subject to their applicable terms.
- NVIDIA: enumerate actual shipped proprietary libraries separately from host driver libraries. Check redistributable portions and component-specific terms at https://docs.nvidia.com/cuda/eula/index.html. The project's NOTICE exception grants only rights held by Dmitry; it is not permission on behalf of Ultralytics or NVIDIA.
- Pexels video: https://www.pexels.com/video/36304379/ and https://www.pexels.com/license/. It retains separate terms, not AGPL. The image checker verifies its pinned byte hash.
- Torch, ONNX Runtime, Triton, nested wheel libraries and Debian packages: installed inventory is now generated, but complete source/notice and redistribution review remains pending.

For each applicable copyleft binary obligation, the release must identify exact source, modifications/build information, delivery method and retention requirements. A link to this application's Git commit does not universally satisfy dependencies' obligations. Full dependency locks/hash installs and repeatable exporter OS packages remain follow-up work; current version pins and inventory do not promise reproducible bytes.
