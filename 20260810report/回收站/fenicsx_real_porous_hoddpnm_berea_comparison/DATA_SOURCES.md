# Real Porous-Medium Data Sources

Primary target for this folder:

- Imperial College London, Pore-Scale Modelling and Imaging, "Berea sandstone".
  The public listing describes micro-CT images and networks and specifically a
  Berea sandstone image/network example.
- Figshare, "Berea Sandstone", DOI `10.6084/m9.figshare.1200118`. The public
  record describes a micro-computed tomography segmented image of Berea
  sandstone.
- Digital Rocks / Digital Porous Media Portal project 317, as indexed by the
  open-source `digital_rocks_data` package. The indexed Berea binary file is
  `Berea_2d25um_binary.raw`, with shape `1000 x 1000 x 1000`, voxel length
  `2.25 um`, `uint8` storage, and download endpoint
  `https://www.digitalrocksportal.org/projects/317/images/223453/download/`.
- Digital Porous Media Portal can also be used as a broader source of published
  porous-media image datasets.

Links:

- https://www.imperial.ac.uk/engineering/departments/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/
- https://www.imperial.ac.uk/engineering/departments/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/berea-sandstone/
- https://figshare.com/articles/dataset/Berea_Sandstone/1200118
- https://digitalporousmedia.org/published-datasets/
- https://github.com/LukasMosser/digital_rocks_data/blob/main/drd/datasets/eleven_sandstones.py

Notes:

- Some figshare/Imperial download endpoints can return HTTP 403 to automated
  clients. If that happens, download the binary segmented image manually in a
  browser and place it under `data/`.
- Keep the original full-resolution file outside git/output artifacts. This
  validation script crops a smaller subvolume at runtime.
- Confirm the phase convention before running. The bundled
  `data/berea_100_to_300.npz` stores pores as `True`/`1`, matching the script
  default `--pore-value 1`. For a different external volume, use
  `--pore-value 0`, `--invert-pore-mask`, `--threshold`, or
  `--pore-below-threshold` only after checking that file's convention.
