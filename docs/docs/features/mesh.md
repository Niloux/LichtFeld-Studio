# Mesh from splats

Extracts a triangle mesh with vertex colors, and optionally a texture, from a trained model. It renders transmittance median
depth from the training cameras, fuses it into a sparse TSDF, extracts the surface with naive
surface nets and colors the vertices from the photos. Each photo counts by 1 / distance, so close
views count more. A model trained with exposure correction or PPISP has each
photo's exposure, vignetting and color taken back out first, so the colors match the model's render
and not any one photo, and photos of different exposure do not leave visible patches. The per-photo
residual grid of exposure correction is not undone. Pixels outside the dataset's masks are not fused. Fisheye, thin-prism, panoramic and rolling-shutter cameras are handled by the same camera
models as training.

Models trained with `--depth-distortion-weight` and `--normal-consistency-weight` give flatter
meshes; fusion cannot remove relief that the splats themselves have.

## Command line

```bash
LichtFeld-Studio mesh project.licht -o mesh.ply
LichtFeld-Studio mesh splat.ply -d dataset -o mesh.ply --voxel-size 0.02
```

With `--texel-size 0.005` it also writes `mesh.obj`, `mesh.mtl` and one `mesh_<page>.jpg` per texture
page, see [Texture](#texture). Writes `mesh.ply` and `mesh.json` with counts and timings. The output folder also receives the
run's `training_config.json` and `metrics.csv`. Held-out views are left out unless `--all-views`
is given. See `LichtFeld-Studio mesh --help`.

## Application

Tools > Mesh from Splats meshes the selected splat from the scene's cameras. It writes a PLY, with
a texel size also a textured OBJ beside it, and adds an undoable mesh node. Training must be stopped, and the complete model must be loaded,
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
| Texel size | 0 | Size of a texture pixel on the surface. 0 keeps vertex colors only |
| Texture size (CLI) | 4096 | Side of a texture page. A scene takes as many pages as it needs |
| Texture width (CLI) | 0 | Render and photo width limit when baking. 0 uses the photos' own width |
| Fill holes (CLI) | 0.3 | Closes holes with a shorter rim using an interpolated patch, not an observed surface. 0 leaves them open |
| Maximum range | 0 | Meshes only surfaces a view saw from closer than this. 0 has no limit. Far surfaces take most of the faces of an outdoor scene at fine voxels, and a pixel there is wider than a voxel |
| Truncation (CLI) | 0.08 or 4 voxels, the larger | Band around the surface. Narrower bands open holes on surfaces seen at grazing angles and where views disagree on depth |
| Minimum component (CLI) | 100 faces | Thin objects can split into small pieces, so keep it low |

GPU memory holds the TSDF, 10 bytes per voxel of each touched block. When it does not fit, fusion
runs in slices of the scene, one at a time. RAM holds one depth map per fused view at the render
width, 1600 pixels by default, and the mesh, so RAM is what limits fine voxels: halving the voxel
size makes about four times the faces. After fusion the job estimates what the rest will take, about
65 bytes per surface voxel, and stops with an error if that exceeds the free RAM. Faces are reduced in
parallel pieces of at most 3 million faces and 4000 tolerances across, 20 m at the default: the
simplifier's single-precision error measure cannot resolve the tolerance over a larger piece and then
keeps most faces. The vertices shared between pieces are kept, a few percent of the output faces.

## Texture

A texel size bakes a texture after faces are reduced, and changes no geometry: the PLY is the same surface with or
without one. Photos of a dataset with `rolling_shutter.json` are placed with their readout motion, on the command line and in
the application alike. xatlas unwraps the mesh into charts on square pages, in groups of about a million neighboring faces with
pages of their own, since packing charts slows sharply with their number. The OBJ repeats only texture coordinates along
chart seams, so its surface stays connected, also in viewers that compute their own normals.

Every texel then takes its color from the photos. Exposure, vignetting and color come out of each photo as they do for
vertex colors. A view counts for a texel only where the four pixels it samples all lie on the surface, so occluding edges
and masked pixels stay out.

Views are chosen in regions, because views never line up to the pixel, and a surface pieced together from several would
break lettering apart. A view rates a face by 1 / distance² and strongly by how square-on it sees it, because an oblique view
slants anything standing off the mesh, such as raised lettering. The view that sees the largest area at no less than half of
each face's best rating takes all of it, then the next view takes what remains, and small islands join the view around them.
Very fine detail can come out softer when the square-on view is the farther one. The chosen view alone sets
everything finer than 1/100 of the image. Broader colors are a mean over all views that see the texel, each counting by
1 / distance, so that neighboring regions agree in level. Lighting that differs between views, such as reflections, can still
show where regions meet.

Texels no view saw take the vertex colors, and the rest of each page continues the charts outward, so that filtering and
image compression reach no background. Baking uses every photo, whatever the view stride: a surface's best view is one or
two photos out of hundreds. It loads each at full size and renders its depth twice, so its time grows with the photos.

RAM holds about 22 bytes per texel and 364 per face, plus about 2 GB while rasterizing 4096 pages, and the job stops with an
error before unwrapping if that exceeds the free RAM. The surface area in `mesh.json` tells how many texels a size takes:
area / size².

`tools/splats_to_mesh.py --mesh mesh.ply` scores a mesh against held-out views and a LAS scan.
