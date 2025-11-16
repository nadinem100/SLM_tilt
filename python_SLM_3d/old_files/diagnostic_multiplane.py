"""
Diagnostic script to analyze multi-plane WGS behavior
Shows exactly which tweezers are in which planes and what z-positions they have
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/mnt/project')
from slm_tweezers_class_WITH_AUTO_CLEANUP import SLMTweezers

# Same parameters as your main script
YAML_PATH = "../slm_parameters.yml"
N_HORIZ = 3
N_VERT = 3
SPACING_UM = 15.0
REDSLM = 1
SCAL = 4
WAIST_UM = 9 / 2 * 1e3
FOCAL_LENGTH_UM = 100000.0
WAVELENGTH_UM = 0.689
TILT_ANGLE_X = 50  # degrees
N_Z_PLANES = 3


def diagnostic_report():
    """Generate comprehensive diagnostic report"""
    
    print("="*80)
    print("MULTI-PLANE WGS DIAGNOSTIC REPORT")
    print("="*80)
    
    # Setup SLM
    print("\n[1] Setting up SLM...")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=1)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)
    
    print(f"    ✓ SLM pixels (reduced): {slm.x_pixels1} × {slm.y_pixels1}")
    print(f"    ✓ Target grid: {N_HORIZ} × {N_VERT} = {N_HORIZ*N_VERT} tweezers")
    print(f"    ✓ Spacing: {SPACING_UM} µm")
    
    # Assign planes
    print(f"\n[2] Assigning planes with tilt_x = {TILT_ANGLE_X}°...")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)
    
    # Extract diagnostic info
    z_per_spot = slm._z_per_spot
    z_planes = slm._z_planes
    members = slm._members
    target_xy_um = slm.target_xy_um
    
    print(f"    ✓ tan({TILT_ANGLE_X}°) = {np.tan(np.deg2rad(TILT_ANGLE_X)):.4f}")
    
    # Z-position analysis
    print("\n" + "="*80)
    print("[3] Z-POSITION ANALYSIS")
    print("="*80)
    
    print(f"\n    Per-tweezer z-offsets:")
    print(f"    {'Index':<8} {'X (µm)':<12} {'Y (µm)':<12} {'Z (µm)':<12}")
    print("    " + "-"*44)
    for i in range(len(z_per_spot)):
        x_um, y_um = target_xy_um[i]
        z_um = z_per_spot[i]
        print(f"    {i:<8} {x_um:>10.2f}   {y_um:>10.2f}   {z_um:>10.2f}")
    
    z_min = np.min(z_per_spot)
    z_max = np.max(z_per_spot)
    z_range = z_max - z_min
    
    print(f"\n    Z-Statistics:")
    print(f"      • Min z:   {z_min:>8.2f} µm")
    print(f"      • Max z:   {z_max:>8.2f} µm")
    print(f"      • Range:   {z_range:>8.2f} µm")
    print(f"      • Mean:    {np.mean(z_per_spot):>8.2f} µm")
    print(f"      • Std:     {np.std(z_per_spot):>8.2f} µm")
    
    # Plane assignment analysis
    print("\n" + "="*80)
    print("[4] PLANE ASSIGNMENT")
    print("="*80)
    
    print(f"\n    Discretized into {N_Z_PLANES} planes:")
    print(f"    {'Plane':<8} {'Z (µm)':<15} {'# Tweezers':<15} {'Tweezer Indices'}")
    print("    " + "-"*70)
    
    for p in range(N_Z_PLANES):
        z_p = z_planes[p]
        n_tweezers = len(members[p])
        indices = members[p]
        print(f"    {p:<8} {z_p:>12.2f}    {n_tweezers:<15} {indices}")
    
    plane_spacing = np.diff(z_planes)
    print(f"\n    Plane spacing: {plane_spacing} µm")
    print(f"    Average spacing: {np.mean(plane_spacing):.2f} µm")
    
    # Expected vs actual focal shift
    print("\n" + "="*80)
    print("[5] EXPECTED FOCAL SHIFT")
    print("="*80)
    
    # For a 50° tilt and 15 µm spacing in X:
    x_span = (N_HORIZ - 1) * SPACING_UM
    expected_z_span = x_span * np.tan(np.deg2rad(TILT_ANGLE_X))
    
    print(f"\n    Grid geometry:")
    print(f"      • X span: {x_span:.2f} µm (from center)")
    print(f"      • Expected z-span for {TILT_ANGLE_X}° tilt: {expected_z_span:.2f} µm")
    print(f"      • Actual z-span: {z_range:.2f} µm")
    print(f"      • Match: {'✓ YES' if abs(expected_z_span - z_range) < 0.1 else '✗ NO'}")
    
    # Visualization range check
    print("\n" + "="*80)
    print("[6] VISUALIZATION RANGE CHECK")
    print("="*80)
    
    viz_range_used = max(z_per_spot)  # What your code uses
    viz_range_needed = max(abs(z_min), abs(z_max)) * 1.5  # Better choice
    
    print(f"\n    Your visualization uses:")
    print(f"      • z_range = max(z_per_spot) = {viz_range_used:.2f} µm")
    print(f"      • This means plotting from -{viz_range_used:.2f} to +{viz_range_used:.2f} µm")
    
    print(f"\n    But your tweezers span:")
    print(f"      • From {z_min:.2f} to {z_max:.2f} µm")
    print(f"      • Suggested viz range: ±{viz_range_needed:.2f} µm")
    
    if abs(viz_range_used) < abs(z_min):
        print(f"\n    ⚠️  WARNING: Your viz range ({viz_range_used:.2f} µm) is SMALLER than")
        print(f"        the minimum z-position ({z_min:.2f} µm)!")
        print(f"        You're not seeing the full tilt effect!")
    
    # Quadratic phase check
    print("\n" + "="*80)
    print("[7] QUADRATIC PHASE CORRECTION")
    print("="*80)
    
    k = 2 * np.pi / WAVELENGTH_UM
    k_over_2f2 = k / (2 * FOCAL_LENGTH_UM**2)
    
    print(f"\n    Optical parameters:")
    print(f"      • k/(2f²) = {k_over_2f2:.6e} µm⁻¹")
    print(f"      • Wavelength: {WAVELENGTH_UM} µm")
    print(f"      • Focal length: {FOCAL_LENGTH_UM} µm")
    
    # For a z-offset of 1 µm, what's the phase at pupil edge?
    pupil_radius_um = slm.params.pixel_um * slm.x_pixels1 / 2
    phase_at_edge = k_over_2f2 * 1.0 * pupil_radius_um**2
    
    print(f"\n    Defocus phase example (1 µm z-offset):")
    print(f"      • Pupil radius: {pupil_radius_um:.2f} µm")
    print(f"      • Phase at edge: {phase_at_edge:.4f} rad = {np.rad2deg(phase_at_edge):.2f}°")
    
    # Create visualization
    print("\n" + "="*80)
    print("[8] CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Top-down view with z-values
    ax = axes[0, 0]
    scatter = ax.scatter(target_xy_um[:, 0], target_xy_um[:, 1], 
                        c=z_per_spot, s=200, cmap='RdYlBu_r', 
                        edgecolors='black', linewidths=2)
    
    # Annotate with indices and z-values
    for i, (x, y, z) in enumerate(zip(target_xy_um[:, 0], target_xy_um[:, 1], z_per_spot)):
        ax.text(x, y, f'{i}\n{z:.1f}', ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('X (µm)', fontsize=12)
    ax.set_ylabel('Y (µm)', fontsize=12)
    ax.set_title(f'Tweezer Positions & Z-Offsets\n(Tilt: {TILT_ANGLE_X}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Z-offset (µm)')
    
    # Plot 2: Side view (X-Z projection)
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, N_Z_PLANES))
    
    for p in range(N_Z_PLANES):
        if members[p]:
            indices = members[p]
            x_vals = target_xy_um[indices, 0]
            z_vals = z_per_spot[indices]
            ax.scatter(x_vals, z_vals, s=200, c=[colors[p]], 
                      label=f'Plane {p} @ z={z_planes[p]:.1f} µm',
                      edgecolors='black', linewidths=2)
            
            # Annotate
            for i, x, z in zip(indices, x_vals, z_vals):
                ax.text(x, z, f'{i}', ha='center', va='center', fontsize=8)
    
    # Draw plane lines
    for p, z_p in enumerate(z_planes):
        ax.axhline(z_p, color=colors[p], linestyle='--', alpha=0.5, linewidth=2)
    
    # Draw expected tilt line
    x_line = np.array([target_xy_um[:, 0].min() - 5, target_xy_um[:, 0].max() + 5])
    x0 = np.mean(target_xy_um[:, 0])
    z_line = np.tan(np.deg2rad(TILT_ANGLE_X)) * (x_line - x0)
    ax.plot(x_line, z_line, 'k-', linewidth=2, label=f'Expected {TILT_ANGLE_X}° tilt')
    
    ax.set_xlabel('X (µm)', fontsize=12)
    ax.set_ylabel('Z-offset (µm)', fontsize=12)
    ax.set_title('Side View: Plane Assignments', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Z-distribution histogram
    ax = axes[1, 0]
    ax.hist(z_per_spot, bins=20, alpha=0.7, edgecolor='black')
    for z_p in z_planes:
        ax.axvline(z_p, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Z-offset (µm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Z-Offset Distribution\n(Red lines = discrete planes)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Grid Setup:
  • {N_HORIZ} × {N_VERT} = {N_HORIZ*N_VERT} tweezers
  • Spacing: {SPACING_UM} µm
  • Tilt angle: {TILT_ANGLE_X}°
  
Z-Range:
  • Min: {z_min:.2f} µm
  • Max: {z_max:.2f} µm  
  • Total span: {z_range:.2f} µm
  • Expected span: {expected_z_span:.2f} µm
  
Planes:
  • Number: {N_Z_PLANES}
  • Spacing: {np.mean(plane_spacing):.2f} µm avg
  
Visualization Issue:
  • Current range: ±{viz_range_used:.2f} µm
  • Needed range: ±{viz_range_needed:.2f} µm
  
{'⚠️  WARNING: Viz range too small!' if abs(viz_range_used) < abs(z_min) else '✓ Viz range OK'}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'debugging/diagnostic_multiplane.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n    ✓ Diagnostic plot saved to: {output_path}")
    
    # Final recommendations
    print("\n" + "="*80)
    print("[9] RECOMMENDATIONS")
    print("="*80)
    
    print("\n    Based on this analysis:")
    
    if abs(z_range) < 1.0:
        print("\n    ⚠️  ISSUE #1: Z-range is very small!")
        print(f"        Your {N_HORIZ}×{N_VERT} grid only spans {z_range:.2f} µm in z.")
        print(f"        With {TILT_ANGLE_X}° tilt, consider:")
        print(f"          • Larger grid (e.g., 5×5 or 7×7)")
        print(f"          • Larger spacing (e.g., 25-30 µm)")
    
    if abs(viz_range_used) < abs(z_min):
        print("\n    ⚠️  ISSUE #2: Visualization range too small!")
        print(f"        Change line in main():")
        print(f"          FROM: z_range_um=max(slm._z_per_spot)")
        print(f"          TO:   z_range_um={viz_range_needed:.1f}")
    
    if N_Z_PLANES > 2 and z_range < 5.0:
        print("\n    ⚠️  ISSUE #3: Too many planes for small z-range!")
        print(f"        With only {z_range:.2f} µm span and {N_Z_PLANES} planes,")
        print(f"        planes are only {np.mean(plane_spacing):.2f} µm apart.")
        print(f"        Consider using 2 planes or increasing grid size.")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    return slm, fig


if __name__ == "__main__":
    slm, fig = diagnostic_report()
    print("\n✓ Check the diagnostic plot to see the full analysis!")
    plt.show()
