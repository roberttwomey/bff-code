#!/usr/bin/env python3
"""
Simple test program for Open3D point cloud visualization
This will help verify if Open3D is working correctly on your system.
"""

import numpy as np
import open3d as o3d
import time

def create_test_point_cloud():
    """Create a simple test point cloud with known geometry."""
    print("Creating test point cloud...")
    
    # Create a simple cube pattern
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    
    # Create grid points
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Filter to create a hollow cube (only surface points)
    mask = ((np.abs(points[:, 0]) >= 1.9) | 
            (np.abs(points[:, 1]) >= 1.9) | 
            (np.abs(points[:, 2]) >= 1.9))
    points = points[mask]
    
    print(f"Created {len(points)} test points")
    return points

def test_basic_visualization():
    """Test basic Open3D visualization."""
    print("Testing basic Open3D visualization...")
    
    # Create test points
    points = create_test_point_cloud()
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color points by position
    colors = np.zeros((len(points), 3))
    colors[:, 0] = (points[:, 0] + 2) / 4  # Red based on X
    colors[:, 1] = (points[:, 1] + 2) / 4  # Green based on Y
    colors[:, 2] = (points[:, 2] + 2) / 4  # Blue based on Z
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Open3D Test - Basic", width=800, height=600)
    vis.add_geometry(pcd)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.7)
    
    print("Window created. You should see a colored cube.")
    print("Press 'q' to close the window.")
    
    # Run visualization
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        vis.destroy_window()
        print("Window closed.")

def test_interactive_visualization():
    """Test interactive Open3D visualization with multiple objects."""
    print("Testing interactive Open3D visualization...")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Open3D Test - Interactive", width=1000, height=700)
    
    # Create multiple test objects
    objects = []
    
    # 1. Point cloud cube
    points = create_test_point_cloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.ones((len(points), 3)) * [1, 0, 0]  # Red
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    objects.append(pcd)
    
    # 2. Sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0, 1, 0])  # Green
    sphere.translate([4, 0, 0])  # Move to the right
    vis.add_geometry(sphere)
    objects.append(sphere)
    
    # 3. Box
    box = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
    box.compute_vertex_normals()
    box.paint_uniform_color([0, 0, 1])  # Blue
    box.translate([-4, 0, 0])  # Move to the left
    vis.add_geometry(box)
    objects.append(box)
    
    # 4. Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)
    objects.append(coord_frame)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)
    
    print("Interactive window created. You should see:")
    print("- Red point cloud cube in center")
    print("- Green sphere on the right")
    print("- Blue box on the left")
    print("- Coordinate frame (RGB axes)")
    print("You can rotate, zoom, and pan with mouse.")
    print("Press 'q' to close the window.")
    
    # Run visualization
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        vis.destroy_window()
        print("Window closed.")

def test_lidar_simulation():
    """Test with simulated lidar-like data."""
    print("Testing lidar-like data visualization...")
    
    # Create simulated lidar data (similar to what we expect from Go2)
    np.random.seed(42)  # For reproducible results
    
    # Ground plane points
    x_ground = np.random.uniform(-10, 10, 1000)
    y_ground = np.random.uniform(-10, 10, 1000)
    z_ground = np.zeros(1000) + np.random.normal(0, 0.1, 1000)
    ground_points = np.column_stack([x_ground, y_ground, z_ground])
    
    # Wall points (vertical surfaces)
    x_walls = np.random.uniform(-10, 10, 500)
    z_walls = np.random.uniform(0, 3, 500)
    y_walls = np.random.choice([-10, 10], 500) + np.random.normal(0, 0.1, 500)
    wall_points = np.column_stack([x_walls, y_walls, z_walls])
    
    # Obstacle points (boxes)
    obstacle_centers = [(-5, -5, 1), (5, 5, 1), (0, 0, 2)]
    obstacle_points = []
    for cx, cy, cz in obstacle_centers:
        x_obs = np.random.uniform(cx-1, cx+1, 200)
        y_obs = np.random.uniform(cy-1, cy+1, 200)
        z_obs = np.random.uniform(cz-0.5, cz+0.5, 200)
        obstacle_points.append(np.column_stack([x_obs, y_obs, z_obs]))
    
    # Combine all points
    all_points = np.vstack([ground_points, wall_points] + obstacle_points)
    print(f"Created {len(all_points)} simulated lidar points")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Color by height (Z coordinate)
    colors = np.zeros((len(all_points), 3))
    z_normalized = (all_points[:, 2] - all_points[:, 2].min()) / (all_points[:, 2].max() - all_points[:, 2].min() + 1e-6)
    colors[:, 0] = z_normalized  # Red for height
    colors[:, 1] = 1 - z_normalized  # Green for inverse height
    colors[:, 2] = 0.3  # Blue constant
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Open3D Test - Lidar Simulation", width=900, height=600)
    vis.add_geometry(pcd)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.1)
    
    print("Lidar simulation window created.")
    print("You should see a room-like environment with:")
    print("- Ground plane (green)")
    print("- Walls (red)")
    print("- Obstacles (mixed colors)")
    print("Press 'q' to close the window.")
    
    # Run visualization
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        vis.destroy_window()
        print("Window closed.")

def main():
    """Run all tests."""
    print("Open3D Point Cloud Visualization Test")
    print("=" * 40)
    
    try:
        print(f"Open3D version: {o3d.__version__}")
        print(f"NumPy version: {np.__version__}")
        print()
        
        # Test 1: Basic visualization
        print("Test 1: Basic Point Cloud Visualization")
        print("-" * 35)
        test_basic_visualization()
        print()
        
        # Test 2: Interactive visualization
        print("Test 2: Interactive Visualization")
        print("-" * 30)
        test_interactive_visualization()
        print()
        
        # Test 3: Lidar simulation
        print("Test 3: Lidar-like Data Simulation")
        print("-" * 35)
        test_lidar_simulation()
        print()
        
        print("All tests completed successfully!")
        print("If you could see all three windows, Open3D is working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
