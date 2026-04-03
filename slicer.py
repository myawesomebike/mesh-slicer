import numpy as np
import trimesh
import pyvista as pv
from scipy.interpolate import splprep, splev
import tkinter as tk
from tkinter import filedialog
import os
import json

# =========================
# CONFIG
# =========================
NUM_SLICES = 30
RESAMPLE_POINTS = 120
SMOOTHING = 0.002

AXIS_MAP = {0: "X", 1: "Y", 2: "Z"}
AXIS_COLORS = {0: "red", 1: "green", 2: "blue"}

# =========================
# FILE SELECTION
# =========================
def select_file():
    """Open file dialog to select STL/OBJ file"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select 3D Mesh File",
        filetypes=[
            ("3D Model Files", "*.stl *.STL *.obj *.OBJ *.glb *.GLB"),
            ("STL Files", "*.stl *.STL"),
            ("OBJ Files", "*.obj *.OBJ"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path

# =========================
# EXPORT FUNCTIONS
# =========================
def export_to_dxf(cross_sections, filepath, axis_index):
    """Export cross sections to DXF format for Fusion 360"""
    try:
        import ezdxf
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        for i, section_data in enumerate(cross_sections):
            layer_name = f"Slice_{i:03d}"
            doc.layers.new(name=layer_name, dxfattribs={'color': 1})
            
            points_3d = section_data['points']
            # Project to 2D based on axis
            if axis_index == 0:  # X-axis
                points_2d = points_3d[:, [1, 2]]  # Y, Z
            elif axis_index == 1:  # Y-axis
                points_2d = points_3d[:, [0, 2]]  # X, Z
            else:  # Z-axis
                points_2d = points_3d[:, [0, 1]]  # X, Y
            
            # Create polyline
            msp.add_lwpolyline(points_2d, dxfattribs={'layer': layer_name, 'closed': True})
        
        doc.saveas(filepath)
        print(f"Exported to DXF: {filepath}")
        return True
    except ImportError:
        print("ezdxf not available, using fallback export")
        return False

def export_to_svg(cross_sections, filepath, axis_index):
    """Export cross sections to SVG format"""
    # Calculate bounds
    all_points = np.vstack([s['points'] for s in cross_sections])
    if axis_index == 0:  # X-axis
        points_2d = all_points[:, [1, 2]]
    elif axis_index == 1:  # Y-axis
        points_2d = all_points[:, [0, 2]]
    else:  # Z-axis
        points_2d = all_points[:, [0, 1]]
    
    min_x, min_y = points_2d.min(axis=0)
    max_x, max_y = points_2d.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Add padding
    padding = max(width, height) * 0.1
    min_x -= padding
    min_y -= padding
    width += 2 * padding
    height += 2 * padding
    
    # Create SVG
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{min_x} {min_y} {width} {height}">',
        f'<g transform="scale(1,-1) translate(0,{-(2*min_y + height)})">'
    ]
    
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, section_data in enumerate(cross_sections):
        points_3d = section_data['points']
        
        if axis_index == 0:  # X-axis
            points_2d = points_3d[:, [1, 2]]
        elif axis_index == 1:  # Y-axis
            points_2d = points_3d[:, [0, 2]]
        else:  # Z-axis
            points_2d = points_3d[:, [0, 1]]
        
        path_data = f"M {points_2d[0,0]},{points_2d[0,1]} "
        for pt in points_2d[1:]:
            path_data += f"L {pt[0]},{pt[1]} "
        path_data += "Z"
        
        color = colors[i % len(colors)]
        svg_lines.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="0.5" id="slice_{i}"/>')
    
    svg_lines.append('</g>')
    svg_lines.append('</svg>')
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Exported to SVG: {filepath}")
    return True

def export_to_json(cross_sections, filepath, axis_index):
    """Export cross sections to JSON format with metadata"""
    export_data = {
        'axis': AXIS_MAP[axis_index],
        'axis_index': int(axis_index),
        'num_sections': len(cross_sections),
        'sections': []
    }
    
    for i, section_data in enumerate(cross_sections):
        export_data['sections'].append({
            'id': i,
            'position': float(section_data['position']),
            'points': section_data['points'].tolist()
        })
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported to JSON: {filepath}")
    return True

# =========================
# HELPERS
# =========================

def resample_closed_curve_3d(points_3d, n=100, smoothing=0.0):
    """Resample a 3D closed curve using spline interpolation"""
    points_3d = np.asarray(points_3d)
    if points_3d.ndim != 2 or points_3d.shape[0] < 8:
        return None
    if points_3d.shape[1] != 3:
        return None
    
    try:
        tck, _ = splprep(points_3d.T, s=smoothing, per=True)
        u_new = np.linspace(0, 1, n)
        resampled = np.array(splev(u_new, tck)).T
        return resampled
    except Exception:
        return None

def largest_loop_3d(section):
    """Extract the largest loop from a 3D path section"""
    if section is None:
        return None
    
    entities = getattr(section, "entities", None)
    if entities is None or len(entities) == 0:
        return None
    
    vertices = section.vertices
    longest_loop = None
    max_length = 0
    
    for entity in entities:
        try:
            if hasattr(entity, "points"):
                points_idx = entity.points
                if len(points_idx) > max_length:
                    max_length = len(points_idx)
                    longest_loop = vertices[points_idx]
        except:
            continue
    
    if longest_loop is not None and len(longest_loop) > 10:
        return longest_loop
    
    return None

# =========================
# LOAD MESH
# =========================
print("=" * 50)
print("3D MESH CROSS-SECTION ANALYZER")
print("=" * 50)

mesh_file = select_file()
if not mesh_file or not os.path.exists(mesh_file):
    print("No file selected. Exiting.")
    exit()

print(f"\nLoading mesh: {os.path.basename(mesh_file)}")
mesh = trimesh.load(mesh_file)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump().sum()

mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()
print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Get mesh bounds
mesh_bounds = mesh.bounds
print(f"Mesh bounds: X({mesh_bounds[0,0]:.2f}, {mesh_bounds[1,0]:.2f}), "
      f"Y({mesh_bounds[0,1]:.2f}, {mesh_bounds[1,1]:.2f}), "
      f"Z({mesh_bounds[0,2]:.2f}, {mesh_bounds[1,2]:.2f})")

# =========================
# PLOTTER SETUP
# =========================
plotter = pv.Plotter()
plotter.set_background("white")

# Add mesh
pv_mesh = pv.wrap(mesh)
plotter.add_mesh(pv_mesh, color="lightblue", opacity=0.4, smooth_shading=True)

# Store actors and cross-section data
sections_actor = []
info_text_actor = None
bbox_actor = None
stored_cross_sections = []

# =========================
# UPDATE FUNCTION
# =========================
def update_sections(axis_slider_val, num_slices_val, smoothing_val, resample_points_val, 
                   bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y, 
                   bbox_min_z, bbox_max_z, show_bbox, update_camera=False):
    global sections_actor, info_text_actor, bbox_actor, stored_cross_sections

    # Remove old cross sections
    for actor in sections_actor:
        plotter.remove_actor(actor)
    sections_actor = []
    
    # Remove old info text
    if info_text_actor is not None:
        plotter.remove_actor(info_text_actor)
    
    # Remove old bounding box
    if bbox_actor is not None:
        plotter.remove_actor(bbox_actor)
        bbox_actor = None

    axis_index = int(round(axis_slider_val))
    axis_name = AXIS_MAP[axis_index]
    axis_color = AXIS_COLORS[axis_index]
    
    # Create bounding box
    bbox = np.array([
        [bbox_min_x, bbox_min_y, bbox_min_z],
        [bbox_max_x, bbox_max_y, bbox_max_z]
    ])
    
    # Show bounding box if enabled
    if show_bbox > 0.5:
        bbox_mesh = pv.Box(bounds=[bbox_min_x, bbox_max_x, 
                                   bbox_min_y, bbox_max_y, 
                                   bbox_min_z, bbox_max_z])
        bbox_actor = plotter.add_mesh(bbox_mesh, style='wireframe', 
                                      color='black', line_width=2, opacity=0.8)
    
    # Use bounding box for slicing range
    slice_min = bbox[0, axis_index]
    slice_max = bbox[1, axis_index]
    
    if slice_max <= slice_min:
        slice_min, slice_max = mesh_bounds[:, axis_index]
    
    slice_positions = np.linspace(slice_min + 1e-3, slice_max - 1e-3, int(num_slices_val))

    sections_created = 0
    stored_cross_sections = []  # Reset stored sections
    
    for pos in slice_positions:
        # Only slice if position is within bounding box
        if pos < bbox[0, axis_index] or pos > bbox[1, axis_index]:
            continue
            
        origin = [0,0,0]; origin[axis_index]=pos
        normal = [0,0,0]; normal[axis_index]=1
        
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            continue
        
        loop_3d = largest_loop_3d(section)
        if loop_3d is None or len(loop_3d) < 10:
            continue
        
        # Filter points to only those within bounding box
        within_bbox = (
            (loop_3d[:, 0] >= bbox_min_x) & (loop_3d[:, 0] <= bbox_max_x) &
            (loop_3d[:, 1] >= bbox_min_y) & (loop_3d[:, 1] <= bbox_max_y) &
            (loop_3d[:, 2] >= bbox_min_z) & (loop_3d[:, 2] <= bbox_max_z)
        )
        
        # If not enough points in bbox, skip this section
        if np.sum(within_bbox) < 10:
            continue
        
        # Only use points within the bounding box
        loop_3d_filtered = loop_3d[within_bbox]
        
        resampled_3d = resample_closed_curve_3d(loop_3d_filtered, n=int(resample_points_val), smoothing=smoothing_val)
        if resampled_3d is None:
            continue
        
        loop_3d_closed = np.vstack([resampled_3d, resampled_3d[0]])

        # Store cross-section data for export
        stored_cross_sections.append({
            'position': pos,
            'points': resampled_3d.copy()  # Don't include the duplicate closing point
        })

        lines = np.arange(loop_3d_closed.shape[0], dtype=np.int_)
        cells = np.hstack([np.array([2, i, i+1]) for i in range(len(lines)-1)])
        poly = pv.PolyData()
        poly.points = loop_3d_closed
        poly.lines = cells

        actor = plotter.add_mesh(poly, color=axis_color, line_width=3, opacity=0.8)
        sections_actor.append(actor)
        sections_created += 1

    # Add info text
    info_text = f"Slicing along {axis_name}-axis\n{sections_created} cross-sections\nPress 'e' to export"
    info_text_actor = plotter.add_text(
        info_text,
        position='upper_right',
        font_size=12,
        color='black'
    )

    # Only update camera if explicitly requested (initial load)
    if update_camera:
        camera_views = {
            0: ([1, 0, 0], [0, 0, 0], [0, 0, 1]),   # X-axis: side view
            1: ([0, 1, 0], [0, 0, 0], [0, 0, 1]),   # Y-axis: front view
            2: ([0, 0, 1], [0, 0, 0], [0, 1, 0])    # Z-axis: top view
        }
        
        if axis_index in camera_views:
            position, focal_point, viewup = camera_views[axis_index]
            plotter.camera_position = 'iso'
            plotter.view_vector(position, viewup=viewup)
            plotter.camera.zoom(1.2)

# =========================
# EXPORT HANDLER
# =========================
def export_cross_sections():
    """Handle export of cross sections"""
    if len(stored_cross_sections) == 0:
        print("No cross-sections to export!")
        return
    
    root = tk.Tk()
    root.withdraw()
    
    # Get base filename
    base_path = filedialog.asksaveasfilename(
        title="Export Cross Sections",
        defaultextension=".dxf",
        filetypes=[
            ("DXF Files", "*.dxf"),
            ("SVG Files", "*.svg"),
            ("JSON Files", "*.json"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    if not base_path:
        print("Export cancelled")
        return
    
    base_name, ext = os.path.splitext(base_path)
    axis_index = state['axis']
    
    # Export in the selected format
    if ext.lower() == '.svg':
        export_to_svg(stored_cross_sections, base_path, axis_index)
    elif ext.lower() == '.json':
        export_to_json(stored_cross_sections, base_path, axis_index)
    elif ext.lower() == '.dxf':
        success = export_to_dxf(stored_cross_sections, base_path, axis_index)
        if not success:
            # Fallback to SVG
            svg_path = base_name + '.svg'
            export_to_svg(stored_cross_sections, svg_path, axis_index)
    else:
        # Default to SVG
        svg_path = base_name + '.svg'
        export_to_svg(stored_cross_sections, svg_path, axis_index)
    
    # Also export JSON with metadata
    json_path = base_name + '_data.json'
    export_to_json(stored_cross_sections, json_path, axis_index)
    
    print(f"\nExport complete!")
    print(f"Total sections exported: {len(stored_cross_sections)}")

# =========================
# STATE VARIABLES
# =========================
state = {
    'axis': 2,
    'num_slices': NUM_SLICES,
    'smoothing': SMOOTHING,
    'resample_points': RESAMPLE_POINTS,
    'bbox_min_x': mesh_bounds[0, 0],
    'bbox_max_x': mesh_bounds[1, 0],
    'bbox_min_y': mesh_bounds[0, 1],
    'bbox_max_y': mesh_bounds[1, 1],
    'bbox_min_z': mesh_bounds[0, 2],
    'bbox_max_z': mesh_bounds[1, 2],
    'show_bbox': 1
}

def make_callback(param_name):
    def callback(value):
        state[param_name] = value
        update_sections(
            state['axis'],
            state['num_slices'],
            state['smoothing'],
            state['resample_points'],
            state['bbox_min_x'],
            state['bbox_max_x'],
            state['bbox_min_y'],
            state['bbox_max_y'],
            state['bbox_min_z'],
            state['bbox_max_z'],
            state['show_bbox'],
            update_camera=False  # Don't update camera on slider changes
        )
    return callback

# =========================
# SLIDER LAYOUT
# =========================
slider_style = {
    'title_height': 0.02,
    'title_opacity': 1.0,
    'title_color': 'black',
    'tube_width': 0.005,
    'slider_width': 0.02,
    'color': 'grey'
}

# Left side panel - Bounding Box Controls
x_left = 0.02
slider_width_left = 0.22
y_positions_left = [0.95, 0.88, 0.78, 0.68, 0.58, 0.48, 0.38]

plotter.add_slider_widget(
    make_callback('show_bbox'),
    [0, 1],
    value=state['show_bbox'],
    title="Show Bbox (0=Off, 1=On)",
    pointa=(x_left, y_positions_left[0]),
    pointb=(x_left + slider_width_left, y_positions_left[0]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_min_x'),
    [mesh_bounds[0, 0], mesh_bounds[1, 0]],
    value=state['bbox_min_x'],
    title="X Min",
    pointa=(x_left, y_positions_left[1]),
    pointb=(x_left + slider_width_left, y_positions_left[1]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_max_x'),
    [mesh_bounds[0, 0], mesh_bounds[1, 0]],
    value=state['bbox_max_x'],
    title="X Max",
    pointa=(x_left, y_positions_left[2]),
    pointb=(x_left + slider_width_left, y_positions_left[2]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_min_y'),
    [mesh_bounds[0, 1], mesh_bounds[1, 1]],
    value=state['bbox_min_y'],
    title="Y Min",
    pointa=(x_left, y_positions_left[3]),
    pointb=(x_left + slider_width_left, y_positions_left[3]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_max_y'),
    [mesh_bounds[0, 1], mesh_bounds[1, 1]],
    value=state['bbox_max_y'],
    title="Y Max",
    pointa=(x_left, y_positions_left[4]),
    pointb=(x_left + slider_width_left, y_positions_left[4]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_min_z'),
    [mesh_bounds[0, 2], mesh_bounds[1, 2]],
    value=state['bbox_min_z'],
    title="Z Min",
    pointa=(x_left, y_positions_left[5]),
    pointb=(x_left + slider_width_left, y_positions_left[5]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('bbox_max_z'),
    [mesh_bounds[0, 2], mesh_bounds[1, 2]],
    value=state['bbox_max_z'],
    title="Z Max",
    pointa=(x_left, y_positions_left[6]),
    pointb=(x_left + slider_width_left, y_positions_left[6]),
    **slider_style
)

# Right side panel - Slicing Controls
x_right = 0.72
slider_width_right = 0.25
y_positions_right = [0.85, 0.70, 0.55, 0.40]

plotter.add_slider_widget(
    make_callback('axis'),
    [0, 2],
    value=state['axis'],
    title="Slice Axis (0=X, 1=Y, 2=Z)",
    pointa=(x_right, y_positions_right[0]),
    pointb=(x_right + slider_width_right, y_positions_right[0]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('num_slices'),
    [5, 100],
    value=state['num_slices'],
    title="Number of Slices",
    pointa=(x_right, y_positions_right[1]),
    pointb=(x_right + slider_width_right, y_positions_right[1]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('smoothing'),
    [0, 0.02],
    value=state['smoothing'],
    title="Smoothing Factor",
    pointa=(x_right, y_positions_right[2]),
    pointb=(x_right + slider_width_right, y_positions_right[2]),
    **slider_style
)

plotter.add_slider_widget(
    make_callback('resample_points'),
    [20, 300],
    value=state['resample_points'],
    title="Points per Section",
    pointa=(x_right, y_positions_right[3]),
    pointb=(x_right + slider_width_right, y_positions_right[3]),
    **slider_style
)

# =========================
# KEYBOARD CONTROLS
# =========================
def key_press_callback(obj=None, event=None):
    print("\nExporting cross-sections...")
    export_cross_sections()

plotter.add_key_event('e', key_press_callback)

# =========================
# TITLE
# =========================
plotter.add_text(
    "3D Mesh Cross-Section Analyzer",
    position='upper_left',
    font_size=14,
    color='black',
    font='arial'
)

# =========================
# INITIAL PLOT
# =========================
update_sections(
    state['axis'],
    state['num_slices'],
    state['smoothing'],
    state['resample_points'],
    state['bbox_min_x'],
    state['bbox_max_x'],
    state['bbox_min_y'],
    state['bbox_max_y'],
    state['bbox_min_z'],
    state['bbox_max_z'],
    state['show_bbox'],
    update_camera=True  # Update camera on initial load
)

# =========================
# SHOW
# =========================
plotter.add_axes(
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
    line_width=3,
    color='black'
)

print("\n=== Controls ===")
print("LEFT PANEL - Bounding Box:")
print("  - Toggle bbox visibility")
print("  - Adjust X, Y, Z min/max to focus on specific region")
print("\nRIGHT PANEL - Slicing:")
print("  - Slice Axis: 0=X (red), 1=Y (green), 2=Z (blue)")
print("\nKEYBOARD:")
print("  - Press 'e' to EXPORT cross-sections")
print("\nMouse Controls:")
print("  - Rotate: Left mouse drag")
print("  - Pan: Middle mouse drag")
print("  - Zoom: Mouse wheel")
print("\n" + "=" * 50)

plotter.show()