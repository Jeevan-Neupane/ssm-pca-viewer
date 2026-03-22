import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

class AdvancedSSMVisualizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.particles = np.loadtxt(os.path.join(data_dir, 'particles.particles'))
        self.mean_shape = self._load_obj(os.path.join(data_dir, 'mean.obj'))
        self.eigenvalues = np.loadtxt(os.path.join(data_dir, 'eigen_values.eval'))
        self.eigenvectors = [np.loadtxt(os.path.join(data_dir, f'eignen_vector{i}.eval')) 
                            for i in range(9)]
    
    def _load_obj(self, path):
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices)
    
    def reconstruct(self, weights):
        shape = self.mean_shape.copy()
        for i, w in enumerate(weights):
            shape += w * np.sqrt(self.eigenvalues[i]) * self.eigenvectors[i]
        return shape
    
    def animate_mode(self, mode_idx=0, save_path=None):
        """Animate shape variation along a specific mode"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Animation parameters
        frames = 60
        std_values = np.concatenate([
            np.linspace(0, 2, frames//2),
            np.linspace(2, -2, frames//2),
            np.linspace(-2, 0, frames//2)
        ])
        
        def update(frame):
            ax.clear()
            weights = np.zeros(len(self.eigenvectors))
            weights[mode_idx] = std_values[frame % len(std_values)]
            shape = self.reconstruct(weights)
            
            ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                      c=shape[:, 2], cmap='viridis', marker='.', s=10, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Mode {mode_idx} Animation (σ = {std_values[frame % len(std_values)]:.2f})')
            ax.view_init(elev=20, azim=frame*2)
        
        anim = FuncAnimation(fig, update, frames=len(std_values), interval=50)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"✓ Animation saved: {save_path}")
        
        return fig, anim
    
    def compare_shapes(self):
        """Compare mean shape with extreme variations"""
        fig = plt.figure(figsize=(15, 5))
        
        # Mean shape
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(self.mean_shape[:, 0], self.mean_shape[:, 1], self.mean_shape[:, 2],
                   c='blue', marker='.', s=10, alpha=0.5)
        ax1.set_title('Mean Shape')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # +2σ variation (Mode 0)
        ax2 = fig.add_subplot(132, projection='3d')
        weights = np.zeros(len(self.eigenvectors))
        weights[0] = 2.0
        shape_plus = self.reconstruct(weights)
        ax2.scatter(shape_plus[:, 0], shape_plus[:, 1], shape_plus[:, 2],
                   c='red', marker='.', s=10, alpha=0.5)
        ax2.set_title('Mode 0: +2σ')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # -2σ variation (Mode 0)
        ax3 = fig.add_subplot(133, projection='3d')
        weights[0] = -2.0
        shape_minus = self.reconstruct(weights)
        ax3.scatter(shape_minus[:, 0], shape_minus[:, 1], shape_minus[:, 2],
                   c='green', marker='.', s=10, alpha=0.5)
        ax3.set_title('Mode 0: -2σ')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        return fig
    
    def plot_correspondence(self):
        """Visualize particle correspondence"""
        fig = plt.figure(figsize=(12, 5))
        
        # 2D projections
        ax1 = fig.add_subplot(121)
        ax1.scatter(self.particles[:, 0], self.particles[:, 1], 
                   c=range(len(self.particles)), cmap='rainbow', s=30, alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Particle Correspondence (XY Projection)')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(122)
        ax2.scatter(self.particles[:, 0], self.particles[:, 2], 
                   c=range(len(self.particles)), cmap='rainbow', s=30, alpha=0.7)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Particle Correspondence (XZ Projection)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_mode_heatmap(self):
        """Heatmap of eigenvector magnitudes"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create matrix of eigenvector magnitudes
        n_modes = len(self.eigenvectors)
        n_points = len(self.eigenvectors[0])
        
        magnitudes = np.zeros((n_modes, n_points))
        for i, ev in enumerate(self.eigenvectors):
            magnitudes[i] = np.linalg.norm(ev, axis=1)
        
        im = ax.imshow(magnitudes, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Mode')
        ax.set_title('Eigenvector Magnitude Heatmap')
        plt.colorbar(im, ax=ax, label='Magnitude')
        plt.tight_layout()
        return fig
    
    def plot_3d_comparison_grid(self):
        """3D grid showing multiple mode combinations"""
        fig = plt.figure(figsize=(15, 10))
        
        modes = [0, 1]
        std_values = [-1.5, 0, 1.5]
        
        plot_idx = 1
        for i, std1 in enumerate(std_values):
            for j, std2 in enumerate(std_values):
                ax = fig.add_subplot(3, 3, plot_idx, projection='3d')
                weights = np.zeros(len(self.eigenvectors))
                weights[modes[0]] = std1
                weights[modes[1]] = std2
                shape = self.reconstruct(weights)
                
                ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2],
                          c='purple', marker='.', s=5, alpha=0.5)
                ax.set_title(f'M{modes[0]}:{std1:.1f}σ, M{modes[1]}:{std2:.1f}σ')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.view_init(elev=20, azim=45)
                plot_idx += 1
        
        plt.tight_layout()
        return fig


def main():
    data_dir = r'c:\Users\jeevan\OneDrive\Desktop\self_ssm'
    viz = AdvancedSSMVisualizer(data_dir)
    
    print("Advanced SSM Visualization")
    print("=" * 60)
    
    # 1. Shape comparison
    print("Creating shape comparison plot...")
    fig1 = viz.compare_shapes()
    fig1.savefig(os.path.join(data_dir, 'shape_comparison.png'), dpi=150)
    print("✓ Saved: shape_comparison.png")
    plt.close(fig1)
    
    # 2. Particle correspondence
    print("Creating correspondence plot...")
    fig2 = viz.plot_correspondence()
    fig2.savefig(os.path.join(data_dir, 'correspondence.png'), dpi=150)
    print("✓ Saved: correspondence.png")
    plt.close(fig2)
    
    # 3. Mode heatmap
    print("Creating mode heatmap...")
    fig3 = viz.plot_mode_heatmap()
    fig3.savefig(os.path.join(data_dir, 'mode_heatmap.png'), dpi=150)
    print("✓ Saved: mode_heatmap.png")
    plt.close(fig3)
    
    # 4. 3D comparison grid
    print("Creating 3D comparison grid...")
    fig4 = viz.plot_3d_comparison_grid()
    fig4.savefig(os.path.join(data_dir, '3d_comparison_grid.png'), dpi=150)
    print("✓ Saved: 3d_comparison_grid.png")
    plt.close(fig4)
    
    # 5. Animation (optional - uncomment to create)
    # print("Creating animation for Mode 0...")
    # fig5, anim = viz.animate_mode(mode_idx=0, 
    #                               save_path=os.path.join(data_dir, 'mode_0_animation.gif'))
    # plt.close(fig5)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
