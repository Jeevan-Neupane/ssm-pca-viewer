import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
from config import DATA_DIR, OUTPUT_DIR

class SSMVisualizer:
    def __init__(self, data_dir, dataset_suffix=''):
        self.data_dir = data_dir
        self.dataset_suffix = dataset_suffix
        self.particles = self.load_particles()
        self.mean_shape = self.load_mean_shape()
        self.eigenvalues = self.load_eigenvalues()
        self.eigenvectors = self.load_eigenvectors()
        
    def load_particles(self):
        """Load particle positions from .particles file"""
        # Try with suffix first, fall back to default
        path = os.path.join(self.data_dir, f'particles{self.dataset_suffix}.particles')
        if not os.path.exists(path):
            path = os.path.join(self.data_dir, 'particles.particles')
        return np.loadtxt(path)
    
    def load_mean_shape(self):
        """Load mean shape from .obj or .vtk file"""
        # Try .vtk first if suffix provided
        if self.dataset_suffix:
            vtk_path = os.path.join(self.data_dir, f'mean{self.dataset_suffix}.vtk')
            if os.path.exists(vtk_path):
                return self._load_vtk(vtk_path)
        
        # Fall back to .obj
        obj_path = os.path.join(self.data_dir, 'mean.obj')
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices)
    
    def _load_vtk(self, path):
        """Load vertices from VTK file"""
        vertices = []
        with open(path, 'r') as f:
            reading_points = False
            for line in f:
                if 'POINTS' in line:
                    reading_points = True
                    continue
                if reading_points:
                    if line.strip() and not line.startswith('POLYGONS'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except ValueError:
                                continue
                    else:
                        break
        return np.array(vertices)
    
    def load_eigenvalues(self):
        """Load eigenvalues from .eval file"""
        path = os.path.join(self.data_dir, f'eigen_values{self.dataset_suffix}.eval')
        return np.loadtxt(path)
    
    def load_eigenvectors(self):
        """Load all eigenvectors"""
        eigenvectors = []
        
        # Determine naming pattern based on dataset
        if self.dataset_suffix == '2':
            # Dataset 2: eigen_vector20, 21, 22, ..., 29, 210, 211, ..., 214
            import glob
            pattern = os.path.join(self.data_dir, 'eigen_vector2*.eval')
            files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).replace('eigen_vector2', '').replace('.eval', '')))
            for file in files:
                eigenvectors.append(np.loadtxt(file))
        else:
            # Dataset 1: eignen_vector0, 1, 2, ..., 8
            i = 0
            while True:
                path = os.path.join(self.data_dir, f'eignen_vector{i}.eval')
                if not os.path.exists(path):
                    break
                eigenvectors.append(np.loadtxt(path))
                i += 1
        
        return eigenvectors
    
    def reconstruct_shape(self, weights):
        """Reconstruct shape using PCA weights"""
        shape = self.mean_shape.copy()
        for i, w in enumerate(weights):
            if i < len(self.eigenvectors):
                shape += w * np.sqrt(self.eigenvalues[i]) * self.eigenvectors[i]
        return shape
    
    def plot_particles(self):
        """Plot particle cloud"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.particles[:, 0], self.particles[:, 1], self.particles[:, 2], 
                   c='blue', marker='o', s=20, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Particle Distribution')
        plt.tight_layout()
        return fig
    
    def plot_mean_shape(self):
        """Plot mean shape"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.mean_shape[:, 0], self.mean_shape[:, 1], self.mean_shape[:, 2],
                   c='red', marker='.', s=10, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mean Shape')
        plt.tight_layout()
        return fig
    
    def plot_eigenvalues(self):
        """Plot eigenvalue spectrum"""
        fig, ax = plt.subplots(figsize=(10, 6))
        variance_explained = self.eigenvalues / np.sum(self.eigenvalues) * 100
        cumulative_variance = np.cumsum(variance_explained)
        
        ax.bar(range(len(self.eigenvalues)), variance_explained, alpha=0.7, label='Individual')
        ax.plot(range(len(self.eigenvalues)), cumulative_variance, 'r-o', label='Cumulative')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title('PCA Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def interactive_shape_explorer(self):
        """Interactive shape variation explorer with ALL modes"""
        num_modes = len(self.eigenvectors)
        
        # For many modes, use scrollable figure or split into columns
        if num_modes > 10:
            # Use two columns for sliders
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.subplots_adjust(bottom=0.55, left=0.05, right=0.95)
            
            # Initial shape
            weights = np.zeros(num_modes)
            shape = self.reconstruct_shape(weights)
            scatter = ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                                c='green', marker='.', s=10, alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Interactive Shape Explorer - {num_modes} Modes Available')
            
            # Create sliders in two columns
            sliders = []
            col1_modes = (num_modes + 1) // 2
            
            for i in range(num_modes):
                if i < col1_modes:
                    # Left column
                    slider_bottom = 0.48 - (i * 0.035)
                    ax_slider = plt.axes([0.08, slider_bottom, 0.35, 0.02])
                else:
                    # Right column
                    slider_bottom = 0.48 - ((i - col1_modes) * 0.035)
                    ax_slider = plt.axes([0.55, slider_bottom, 0.35, 0.02])
                
                variance_pct = (self.eigenvalues[i] / np.sum(self.eigenvalues) * 100)
                slider = Slider(ax_slider, f'M{i} ({variance_pct:.1f}%)', 
                              -3.0, 3.0, valinit=0.0)
                sliders.append(slider)
        else:
            # Single column for fewer modes
            slider_height = 0.025
            slider_spacing = 0.03
            total_slider_space = num_modes * slider_spacing
            bottom_margin = min(0.5, total_slider_space + 0.1)
            
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            plt.subplots_adjust(bottom=bottom_margin, left=0.1, right=0.9)
            
            # Initial shape
            weights = np.zeros(num_modes)
            shape = self.reconstruct_shape(weights)
            scatter = ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                                c='green', marker='.', s=10, alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Interactive Shape Explorer - {num_modes} Modes Available')
            
            # Create sliders for ALL modes
            sliders = []
            for i in range(num_modes):
                slider_bottom = bottom_margin - 0.05 - i * slider_spacing
                ax_slider = plt.axes([0.15, slider_bottom, 0.7, slider_height])
                variance_pct = (self.eigenvalues[i] / np.sum(self.eigenvalues) * 100)
                slider = Slider(ax_slider, f'Mode {i} ({variance_pct:.1f}%)', 
                              -3.0, 3.0, valinit=0.0)
                sliders.append(slider)
        
        def update(val):
            weights = np.array([s.val for s in sliders])
            shape = self.reconstruct_shape(weights)
            scatter._offsets3d = (shape[:, 0], shape[:, 1], shape[:, 2])
            fig.canvas.draw_idle()
        
        for slider in sliders:
            slider.on_changed(update)
        
        return fig
    
    def plot_mode_variations(self, mode_idx=0, num_steps=5):
        """Plot shape variations along a specific mode"""
        fig = plt.figure(figsize=(15, 3))
        std_range = np.linspace(-2, 2, num_steps)
        
        for i, std in enumerate(std_range):
            ax = fig.add_subplot(1, num_steps, i+1, projection='3d')
            weights = np.zeros(len(self.eigenvectors))
            weights[mode_idx] = std
            shape = self.reconstruct_shape(weights)
            
            ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                      c='purple', marker='.', s=5, alpha=0.5)
            ax.set_title(f'{std:.1f}σ')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=45)
        
        fig.suptitle(f'Mode {mode_idx} Variations', fontsize=14)
        plt.tight_layout()
        return fig
    
    def summary_report(self):
        """Print summary statistics"""
        print("=" * 60)
        print("Statistical Shape Model Summary")
        print("=" * 60)
        print(f"Number of particles: {len(self.particles)}")
        print(f"Number of vertices in mean shape: {len(self.mean_shape)}")
        print(f"Number of PCA modes: {len(self.eigenvectors)}")
        print(f"\nEigenvalues: {self.eigenvalues}")
        print(f"\nVariance explained by each mode:")
        variance = self.eigenvalues / np.sum(self.eigenvalues) * 100
        for i, v in enumerate(variance):
            print(f"  Mode {i}: {v:.2f}%")
        print(f"\nCumulative variance (first 3 modes): {np.sum(variance[:3]):.2f}%")
        print("=" * 60)


def main():
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
        dataset_suffix = '2' if 'dataset2' in datasets[idx].lower() else ''
    except ValueError:
        print("Invalid input!")
        return
    
    # Initialize visualizer
    print(f"\nLoading {datasets[idx]}...")
    viz = SSMVisualizer(data_dir, dataset_suffix=dataset_suffix)
    
    # Print summary
    viz.summary_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Particle distribution
    fig1 = viz.plot_particles()
    fig1.savefig(os.path.join(OUTPUT_DIR, f'particles_plot{dataset_suffix}.png'), dpi=150)
    print(f"✓ Saved: particles_plot{dataset_suffix}.png")
    
    # 2. Mean shape
    fig2 = viz.plot_mean_shape()
    fig2.savefig(os.path.join(OUTPUT_DIR, f'mean_shape_plot{dataset_suffix}.png'), dpi=150)
    print(f"✓ Saved: mean_shape_plot{dataset_suffix}.png")
    
    # 3. Eigenvalue spectrum
    fig3 = viz.plot_eigenvalues()
    fig3.savefig(os.path.join(OUTPUT_DIR, f'eigenvalues_plot{dataset_suffix}.png'), dpi=150)
    print(f"✓ Saved: eigenvalues_plot{dataset_suffix}.png")
    
    # 4. Mode variations for first 3 modes
    for mode in range(min(3, len(viz.eigenvectors))):
        fig = viz.plot_mode_variations(mode_idx=mode)
        fig.savefig(os.path.join(OUTPUT_DIR, f'mode_{mode}_variations{dataset_suffix}.png'), dpi=150)
        print(f"✓ Saved: mode_{mode}_variations{dataset_suffix}.png")
        plt.close(fig)
    
    # 5. Interactive explorer (will open in separate window)
    print(f"\nOpening interactive shape explorer with {len(viz.eigenvectors)} sliders...")
    print("Use sliders to explore shape variations!")
    viz.interactive_shape_explorer()
    
    plt.show()


if __name__ == '__main__':
    main()
