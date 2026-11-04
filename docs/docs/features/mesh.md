# Mesh from splats

Extracts a vertex-colored triangle mesh from a trained model. It renders transmittance median
depth from the training cameras, fuses it into a sparse TSDF, extracts the surface with naive
surface nets and colors the vertices from the photos. Pixels outside the dataset's masks are not
fused. Fisheye, thin-prism, panoramic and rolling-shutter cameras are handled by the same camera
models as training.

Models trained with `--depth-distortion-weight` and `--normal-consistency-weight` give flatter
meshes; fusion cannot remove relief that the splats themselves have.

## Command line

```bash
LichtFeld-Studio mesh project.licht -o mesh.ply
LichtFeld-Studio mesh splat.ply -d dataset -o mesh.ply --voxel-size 0.02
```

Writes `mesh.ply` and `mesh.json` with counts and timings. The output folder also receives the
run's `training_config.json` and `metrics.csv`. Held-out views are left out unless `--all-views`
is given. See `LichtFeld-Studio mesh --help`.

## Application

Tools > Mesh from Splats meshes the selected splat from the scene's cameras. It writes a PLY and
adds an undoable mesh node. Training must be stopped, and the complete model must be loaded,
not a level-of-detail tree.

## Settings

Distances are in scene units, and the defaults are tuned on metric indoor scans; scale them for
other scenes.

| Setting | Default | Effect |
|---|---|---|
| Voxel size | 0.02 | Finer voxels stop helping once they pass the depth accuracy, and add noise |
| View stride | 2 | Fuse every Nth view. More views fill holes and roughen surfaces |
| Full-weight range | 1 | Views beyond it count less, by 1/r², so close views win. 0 weighs all equally |
| Minimum views | 3 | Views a voxel needs. Higher removes floaters and opens holes |
| Simplify tolerance | 0.005 | Surface deviation allowed when reducing faces, spending more faces where colors change. 0 keeps every face |
| Fill holes (CLI) | 0.3 | Closes holes with a shorter rim using an interpolated patch, not an observed surface. 0 leaves them open |
| Truncation (CLI) | 0.08 | Band around the surface. Narrower bands open holes where views disagree on depth |
| Minimum component (CLI) | 100 faces | Thin objects can split into small pieces, so keep it low |

GPU memory holds the TSDF, 10 bytes per voxel of each touched block. RAM holds one depth map per
fused view at the render width, 1600 pixels by default.

`tools/splats_to_mesh.py --mesh mesh.ply` scores a mesh against held-out views and a LAS scan.
