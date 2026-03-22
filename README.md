# Statistical Shape Model (SSM) Visualization Tool

This toolkit provides comprehensive visualization capabilities for Statistical Shape Models with particle-based correspondence.

## Data Structure

Your SSM contains:
- **particles.particles**: 3D point cloud (125 particles × 3 coordinates)
- **mean.obj**: Mean shape mesh (3D vertices)
- **eigen_values.eval**: 9 eigenvalues representing variance in each PCA mode
- **eignen_vector0-8.eval**: 9 eigenvectors representing shape variation modes

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

### 1. Basic Visualization (`visualize_ssm.py`)

Run the main visualization script:

```bash
python visualize_ssm.py
```

**Features:**
- Particle distribution plot
- Mean shape visualization
- Eigenvalue spectrum (variance explained)
- Mode variation plots (first 3 modes)
- **Interactive shape explorer** with sliders to explore shape variations

**Outputs:**
- `particles_plot.png` - 3D scatter plot of particle positions
- `mean_shape_plot.png` - Mean shape visualization
- `eigenvalues_plot.png` - PCA variance explained chart
- `mode_0_variations.png` - Shape variations along mode 0
- `mode_1_variations.png` - Shape variations along mode 1
- `mode_2_variations.png` - Shape variations along mode 2
- Interactive window with sliders for real-time exploration

### 2. Advanced Visualization (`advanced_visualize.py`)

Run the advanced visualization script:

```bash
python advanced_visualize.py
```

**Features:**
- Shape comparison (mean vs. extreme variations)
- Particle correspondence visualization
- Eigenvector magnitude heatmap
- 3D comparison grid (multiple mode combinations)
- Animation support (optional)

**Outputs:**
- `shape_comparison.png` - Side-by-side comparison of mean and extreme shapes
- `correspondence.png` - 2D projections showing particle correspondence
- `mode_heatmap.png` - Heatmap of eigenvector magnitudes across all modes
- `3d_comparison_grid.png` - Grid showing combinations of first two modes

### 3. Creating Animations (Optional)

To create animated GIFs showing shape variations, uncomment the animation section in `advanced_visualize.py`:

```python
# Uncomment these lines in the main() function:
print("Creating animation for Mode 0...")
fig5, anim = viz.animate_mode(mode_idx=0, 
                              save_path=os.path.join(data_dir, 'mode_0_animation.gif'))
plt.close(fig5)
```

**Note:** Requires `pillow` package:
```bash
pip install pillow
```

## Understanding the Visualizations

### Eigenvalue Spectrum
- Shows how much variance each PCA mode captures
- First few modes typically capture most variation
- Your data: Mode 0 captures ~52%, Mode 1 ~19%, Mode 2 ~8%

### Mode Variations
- Shows shape changes along each principal component
- Range: -2σ to +2σ (standard deviations)
- Helps understand what anatomical variations each mode represents

### Interactive Explorer
- Use sliders to adjust weights for first 5 modes
- Real-time 3D visualization of reconstructed shape
- Explore shape space interactively

### Correspondence Visualization
- Color-coded particles show point-to-point correspondence
- Same color = corresponding anatomical location across shapes
- Essential for statistical analysis

## Customization

### Change Number of Modes in Interactive Explorer

Edit `visualize_ssm.py`, line ~115:
```python
num_modes = min(5, len(self.eigenvectors))  # Change 5 to desired number
```

### Adjust Variation Range

Edit mode variation range in `plot_mode_variations()`:
```python
std_range = np.linspace(-2, 2, num_steps)  # Change -2, 2 to desired range
```

### Change View Angle

Adjust 3D plot viewing angle:
```python
ax.view_init(elev=20, azim=45)  # Change elevation and azimuth
```

## Programmatic Usage

You can also use the visualizer classes in your own scripts:

```python
from visualize_ssm import SSMVisualizer

# Initialize
viz = SSMVisualizer(r'c:\Users\jeevan\OneDrive\Desktop\self_ssm')

# Print summary
viz.summary_report()

# Reconstruct a shape with custom weights
weights = [1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
custom_shape = viz.reconstruct_shape(weights)

# Create custom visualizations
fig = viz.plot_particles()
plt.show()
```

## Troubleshooting

### Issue: "No module named 'matplotlib'"
**Solution:** Install matplotlib: `pip install matplotlib`

### Issue: Interactive window doesn't appear
**Solution:** Make sure you're not running in a headless environment. Try adding `plt.ion()` before creating plots.

### Issue: Plots look cluttered
**Solution:** Reduce point size in scatter plots by changing the `s` parameter:
```python
ax.scatter(..., s=5)  # Smaller points
```

### Issue: Animation creation is slow
**Solution:** Reduce number of frames or resolution:
```python
frames = 30  # Reduce from 60
anim.save(save_path, writer='pillow', fps=10, dpi=100)
```

## Data Format Details

### Particles File Format
```
x1 y1 z1
x2 y2 z2
...
```

### OBJ File Format
```
v x1 y1 z1
v x2 y2 z2
...
```

### Eigenvalue/Eigenvector Format
```
value1
value2
...
```

## Citation

If you use this visualization tool in your research, please cite your SSM generation method and this tool.

## License

MIT License - Feel free to modify and distribute.

## Contact

For issues or questions, please open an issue on the repository.
