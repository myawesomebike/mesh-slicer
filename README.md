# Mesh Slicer

Python app for creating cross-sections from 3D scan files.

## Features

- Import 3D scan STLs from Polycam and other LIDAR sources
- Crop scene along XYZ
- Create slices along X/Y/Z axis
- Refine curve-fitting, cross-section count, and points-per-curve
- Export DXF, JSON, and SVG for use in Fusion360

## How to Use

Mesh Slicer is designed for scanning 3d objects (such as vehicles) and creating individual sketches along an axis. Sketches can be used in Fusion360 for creating objects by lofting between slices. Each slice sketch can be cleaned up and refined in Fusion360 instead of using meshes.

## Images

![Slicing a scan](https://github.com/myawesomebike/mesh-slicer/blob/main/images/slicing.png)