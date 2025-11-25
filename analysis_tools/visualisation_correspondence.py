import numpy as np
import meshplot as mp
import trimesh


def normalize_colors(vertices):
    """Generate normalized RGB colors based on vertex positions."""
    min_coord = np.min(vertices, axis=0, keepdims=True)
    max_coord = np.max(vertices, axis=0, keepdims=True)
    return (vertices - min_coord) / (max_coord - min_coord + 1e-12)

def apply_rotation(vertices, R):
    """Apply only rotation to vertices."""
    return vertices @ R.T
def apply_transform(vertices, R,t):
    """Apply rotation (R) and translation (t) to vertices."""
    return vertices @ R.T+t   # (N,3) @ (3,3)^T + (3,)

def double_plot_with_correspondences(
    verts_src, faces_src,
    verts_tgt_transformed, faces_tgt,
    colors_src, colors_tgt
):
    """
    Plot two meshes side by side (or overlapping if aligned), with vertex colors.
    No correspondence lines.
    """
    # Plot both meshes in the same viewer
    plot_src = mp.plot(verts_src, faces_src, c=colors_src)
    plot_src.add_mesh(verts_tgt_transformed, faces_tgt, c=colors_tgt)

    return plot_src

def double_plot_side_by_side(
    verts_src, faces_src,
    verts_tgt_rotated_shifted, faces_tgt,
    colors_src, colors_tgt
):
    """
    Plot two meshes side by side (same orientation, different positions) with vertex colors.
    """
    plot = mp.plot(verts_src, faces_src, c=colors_src)
    plot.add_mesh(verts_tgt_rotated_shifted, faces_tgt, c=colors_tgt)
    return plot



def double_plot_with_correspondences_lines(verts_src, faces_src, verts_tgt, faces_tgt,
                                     colors_src, colors_tgt, correspondence, num_lines=100):
    """
    Plot two meshes side by side with vertex colors and colored correspondence lines.
    """
    # First subplot returns a Plot object (can add lines)
    plot_src = mp.subplot(verts_src, faces_src, c=colors_src, s=[2, 2, 0])

    # Second subplot shares the figure
    mp.subplot(verts_tgt, faces_tgt, c=colors_tgt, s=[2, 2, 1], data=plot_src)

    # Pick a subset of correspondences to avoid clutter
    sample_idx = np.random.choice(len(correspondence), size=min(num_lines, len(correspondence)), replace=False)

    # Source and target matched vertices
    src_points = verts_src[sample_idx]
    tgt_points = verts_tgt[correspondence[sample_idx]]

    # Shift target mesh for side-by-side display
    offset = np.max(verts_src[:, 0]) - np.min(verts_tgt[:, 0]) + 1.0
    tgt_points_shifted = tgt_points.copy()
    tgt_points_shifted[:, 0] += offset

    # Line colors from source
    #line_colors = colors_src[sample_idx]

    # Draw lines on the left subplot
    #plot_src.add_lines(src_points, tgt_points_shifted,
    #                  shading={"line_color": line_colors, "line_width": 2})

    return plot_src

def visualize_correspondence_meshplot(source_path, target_path, map_path, R):
    """
    Load meshes and point-to-point map (1-based), visualize color transfer
    """
    # Load meshes
    src_mesh = trimesh.load(source_path)
    tgt_mesh = trimesh.load(target_path)

    verts_src = np.array(src_mesh.vertices)
    faces_src = np.array(src_mesh.faces)
    verts_tgt = np.array(tgt_mesh.vertices)
    faces_tgt = np.array(tgt_mesh.faces)

    # Load mapping (1-based) and convert to 0-based
    correspondence = np.loadtxt(map_path, dtype=int) - 1

    #if np.any(correspondence < 0):
    #    raise ValueError(
    #        " Found negative indices after conversion — "
    #        "check if your map.txt is already 0-based."
    #    )

    # Optional safety check
    #if np.max(correspondence) >= len(verts_tgt):
    #    raise IndexError(
    #        f" Map indices exceed target vertex count: max index {np.max(correspondence)}, "
    #        f"num target vertices {len(verts_tgt)}"
    #    )

    # Apply rotation and translation to target vertices
    #verts_tgt_transformed = apply_transform(verts_tgt, R)
    verts_tgt_rotated = apply_rotation(verts_tgt, R)
    
    # Compute translation to display side-by-side
    x_shift = np.max(verts_src[:, 0]) - np.min(verts_tgt_rotated[:, 0]) + 1.0
    verts_tgt_rotated_shifted = verts_tgt_rotated.copy()
    verts_tgt_rotated_shifted[:, 0] += x_shift
    
    # Assign vertex colors
    colors_src = normalize_colors(verts_src)
    colors_tgt = colors_src[correspondence]

    
    # Plot source and rotated+shifted target
    double_plot_side_by_side(
        verts_src, faces_src,
        verts_tgt_rotated_shifted, faces_tgt,
        colors_src, colors_tgt
    )
    #double_plot_with_correspondences(
    #    verts_src, faces_src,
    #    verts_tgt_transformed, faces_tgt,
    #    colors_src, colors_tgt
    #)
    

