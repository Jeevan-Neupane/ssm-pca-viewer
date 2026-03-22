import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
from config import DATA_DIR, OUTPUT_DIR

class ShapeWorksStyleViewer:
    def __init__(self, data_dir, mean_file='mean.obj'):
        self.data_dir = data_dir
        self.particles = np.loadtxt(os.path.join(data_dir, 'particles.particles'))
        self.mesh_vertices = self._load_mesh(os.path.join(data_dir, mean_file))
        
        # Find eigenvalue file with flexible naming
        import glob
        eval_files = glob.glob(os.path.join(data_dir, '*eigenvalue*.eval'))
        if not eval_files:
            eval_files = glob.glob(os.path.join(data_dir, 'eigen_values*.eval'))
        
        self.eigenvalues = np.loadtxt(eval_files[0])
        self.eigenvectors = self._load_eigenvectors()
        self._build_deformation_map()
    
    def _load_mesh(self, path):
        if path.endswith('.vtk'):
            return self._load_vtk(path)
        else:
            return self._load_obj(path)
    
    def _load_vtk(self, path):
        vertices = []
        reading_points = False
        with open(path, 'r') as f:
            for line in f:
                if 'POINTS' in line:
                    reading_points = True
                    continue
                if reading_points:
                    if line.strip() and not line.startswith('POLYGONS') and not line.startswith('CELLS'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except ValueError:
                                reading_points = False
                    else:
                        reading_points = False
        return np.array(vertices)
    
    def _load_obj(self, path):
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices)
    
    def _load_eigenvectors(self):
        import glob
        import re
        eigenvectors = []
        
        # Find all eigenvector files with any naming pattern
        all_files = glob.glob(os.path.join(self.data_dir, '*.eval'))
        
        # Filter files that contain 'vector' (case insensitive)
        vector_files = [f for f in all_files if 'vector' in os.path.basename(f).lower()]
        
        # Sort by extracting numbers from filenames
        def extract_number(filename):
            # Extract all numbers from filename
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else 0
        
        vector_files.sort(key=extract_number)
        
        for f in vector_files:
            eigenvectors.append(np.loadtxt(f))
        
        return eigenvectors
    
    def _build_deformation_map(self):
        from scipy.spatial import cKDTree
        tree = cKDTree(self.particles)
        distances, indices = tree.query(self.mesh_vertices, k=3)
        
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        self.nearest_particles = indices
        self.interpolation_weights = weights
    
    def reconstruct(self, weights):
        particles_deformed = self.particles.copy()
        for i, w in enumerate(weights):
            particles_deformed += w * np.sqrt(self.eigenvalues[i]) * self.eigenvectors[i]
        
        particle_displacement = particles_deformed - self.particles
        mesh_deformed = self.mesh_vertices.copy()
        
        for i in range(len(self.mesh_vertices)):
            displacement = np.sum(
                particle_displacement[self.nearest_particles[i]] * 
                self.interpolation_weights[i][:, np.newaxis],
                axis=0
            )
            mesh_deformed[i] += displacement
        
        return mesh_deformed
    
    def launch_viewer(self):
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes([0.05, 0.25, 0.9, 0.7], projection='3d')
        
        weights = np.zeros(len(self.eigenvectors))
        shape = self.reconstruct(weights)
        scatter = ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                            c=shape[:, 2], cmap='bone', marker='.', s=1, alpha=0.8)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title('Interactive Shape Viewer - Adjust Eigenvector Weights', fontsize=12, pad=20)
        ax.set_facecolor('white')
        
        sliders = []
        n_modes = len(self.eigenvectors)
        variance = self.eigenvalues / np.sum(self.eigenvalues) * 100
        
        for i in range(n_modes):
            col = i % 2
            row = i // 2
            left = 0.08 + col * 0.5
            bottom = 0.18 - row * 0.025
            
            ax_slider = plt.axes([left, bottom, 0.35, 0.015])
            slider = Slider(
                ax_slider, 
                f'Mode {i} ({variance[i]:.1f}%)', 
                -2.0, 2.0, 
                valinit=0.0,
                valstep=0.1
            )
            sliders.append(slider)
        
        def update(val):
            weights = np.array([s.val for s in sliders])
            shape = self.reconstruct(weights)
            scatter._offsets3d = (shape[:, 0], shape[:, 1], shape[:, 2])
            scatter.set_array(shape[:, 2])
            fig.canvas.draw_idle()
        
        for slider in sliders:
            slider.on_changed(lambda val: None)
            slider.valtext.set_text('{:.1f}'.format(slider.val))
            
            def make_release_handler(s):
                def on_release(event):
                    update(s.val)
                return on_release
            
            slider.ax.figure.canvas.mpl_connect('button_release_event', 
                                                 make_release_handler(slider))
        
        from matplotlib.widgets import Button
        ax_reset = plt.axes([0.45, 0.01, 0.1, 0.03])
        btn_reset = Button(ax_reset, 'Reset')
        
        def reset(event):
            for slider in sliders:
                slider.reset()
        
        btn_reset.on_clicked(reset)
        plt.show()


def main():
    import glob
    
    # Discover available datasets
    datasets = [d for d in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not datasets:
        print("Error: No datasets found in data folder!")
        return
    
    print("Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(datasets):
            print("Invalid choice!")
            return
        data_dir = os.path.join(DATA_DIR, datasets[idx])
    except ValueError:
        print("Invalid input!")
        return
    
    mean_files = glob.glob(os.path.join(data_dir, 'mean*.obj')) + \
                 glob.glob(os.path.join(data_dir, 'mean*.vtk'))
    
    if not mean_files:
        print("Error: No mean shape files found!")
        return
    
    mean_file = os.path.basename(mean_files[0])
    
    print("\nLoading mesh and building deformation map...")
    viewer = ShapeWorksStyleViewer(data_dir, mean_file)
    print(f"Loaded: {len(viewer.mesh_vertices)} vertices")
    print(f"Using: {len(viewer.particles)} particles")
    print(f"Found: {len(viewer.eigenvectors)} modes")
    print("Controls:")
    print("  - Drag sliders to adjust modes")
    print("  - Rendering on slider release")
    print("  - Click 'Reset' to return to mean")
    print("=" * 60)
    
    viewer.launch_viewer()


if __name__ == '__main__':
    main()
