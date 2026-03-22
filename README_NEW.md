# Statistical Shape Model (SSM) Visualization Tool

## Project Structure

```
self_ssm/
├── data/
│   ├── dataset1/          # Dataset 1 (9 eigenvectors)
│   │   ├── particles.particles
│   │   ├── mean.obj
│   │   ├── eigen_values.eval
│   │   └── eignen_vector0-8.eval
│   └── dataset2/          # Dataset 2 (15 eigenvectors)
│       ├── mean2.vtk
│       ├── eigen_values2.eval
│       └── eigen_vector20-214.eval
├── src/
│   ├── config.py          # Configuration and paths
│   ├── visualize_ssm.py   # Main visualization script
│   ├── shapeworks_viewer.py  # Interactive viewer
│   └── advanced_visualize.py # Advanced visualizations
├── outputs/               # Generated plots and visualizations
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Viewer (Recommended)

```bash
cd src
python shapeworks_viewer.py
```

Select dataset (1 or 2) and explore shapes interactively with sliders.

### Generate All Visualizations

```bash
cd src
python visualize_ssm.py
```

Outputs saved to `outputs/` directory.

## Features

- **Interactive shape explorer** with real-time sliders
- Particle distribution visualization
- Mean shape visualization
- Eigenvalue spectrum analysis
- Mode variation plots
- Render-on-release for better performance

## Controls

- Drag sliders to adjust shape modes
- Rendering happens when slider is released (not during drag)
- Click 'Reset' to return to mean shape
- Rotate 3D view by clicking and dragging
