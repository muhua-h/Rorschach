import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import random
from PIL import Image, ImageFilter, ImageDraw
import io


class InkblotGenerator:
    def __init__(self, width=600, height=400):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

    def generate_organic_shape(self, center_x, center_y, size, num_points):
        """Generate points for an organic, blob-like shape"""
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * np.pi
            radius_variation = random.uniform(0.4, 1.0)
            radius = size * radius_variation

            # Add noise for organic feel
            noise_x = random.uniform(-15, 15)
            noise_y = random.uniform(-15, 15)

            x = center_x + np.cos(angle) * radius + noise_x
            y = center_y + np.sin(angle) * radius + noise_y
            points.append([x, y])

        return np.array(points)

    def create_inkblot(self, use_color=False, complexity=3, bizarre_mode=False):
        """Generate a complete inkblot image"""
        # Create figure and axis
        fig, ax = plt.subplots(1, 1,
                               figsize=(self.width / 100, self.height / 100),
                               dpi=100)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('#f8f9fa')

        # Color palette for colored mode
        colors = [
            '#8B0000', '#4B0082', '#2F4F4F', '#556B2F', '#8B4513',
            '#800080', '#483D8B', '#2E8B57', '#B8860B', '#CD853F'
        ]

        # Calculate number of blobs based on complexity
        num_blobs = int(complexity * 1.5) + 1

        all_shapes = []  # Store shapes for mirroring

        for blob_idx in range(num_blobs):
            # Choose color
            if use_color:
                color = random.choice(colors)
                alpha = 0.8 if blob_idx == 0 else 0.6  # First blob more opaque
            else:
                color = '#000000'
                alpha = 0.9

            # Random position (left half only unless bizarre mode)
            max_x = self.width - 100 if bizarre_mode else self.center_x - 100
            blob_center_x = random.uniform(50, max_x)
            blob_center_y = random.uniform(50, self.height - 50)
            blob_size = random.uniform(30, 90) * (complexity * 0.3 + 0.4)

            # Generate main blob shape
            num_points = int(complexity * 3 + 8)
            shape_points = self.generate_organic_shape(blob_center_x,
                                                       blob_center_y, blob_size,
                                                       num_points)

            # Create polygon
            polygon = Polygon(shape_points, facecolor=color, alpha=alpha,
                              edgecolor='none')
            ax.add_patch(polygon)

            # Store shape for mirroring (if not bizarre mode)
            if not bizarre_mode:
                all_shapes.append(('polygon', shape_points, color, alpha))

            # Add secondary smaller blobs based on complexity
            secondary_chance = complexity * 0.2
            if random.random() < secondary_chance:
                secondary_count = int(complexity * 0.5) + 1

                for s in range(secondary_count):
                    small_x = blob_center_x + random.uniform(-40, 40)
                    small_y = blob_center_y + random.uniform(-40, 40)
                    small_size = random.uniform(5, 20) * (
                                complexity * 0.2 + 0.6)

                    # Create small circular blob
                    circle = plt.Circle((small_x, small_y), small_size,
                                        facecolor=color, alpha=alpha * 0.8,
                                        edgecolor='none')
                    ax.add_patch(circle)

                    # Store for mirroring
                    if not bizarre_mode:
                        all_shapes.append(('circle', small_x, small_y,
                                           small_size, color, alpha * 0.8))

        # Mirror shapes to right side (unless bizarre mode)
        if not bizarre_mode:
            for shape_data in all_shapes:
                shape_type = shape_data[0]

                if shape_type == 'circle':
                    # Mirror circle
                    _, orig_x, orig_y, size, color, alpha = shape_data
                    mirrored_x = self.width - orig_x
                    circle = plt.Circle((mirrored_x, orig_y), size,
                                        facecolor=color, alpha=alpha,
                                        edgecolor='none')
                    ax.add_patch(circle)
                elif shape_type == 'polygon':
                    # Mirror polygon
                    _, points, color, alpha = shape_data
                    mirrored_points = points.copy()
                    mirrored_points[:, 0] = self.width - points[:,
                                                         0]  # Mirror x coordinates

                    polygon = Polygon(mirrored_points, facecolor=color,
                                      alpha=alpha, edgecolor='none')
                    ax.add_patch(polygon)

        # Add texture effect
        texture_points = complexity * 8
        for i in range(texture_points):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(0.5, 2)

            circle = plt.Circle((x, y), size, facecolor='black', alpha=0.1,
                                edgecolor='none')
            ax.add_patch(circle)

        plt.tight_layout()
        return fig

    def save_inkblot(self, filename, use_color=False, complexity=3,
                     bizarre_mode=False):
        """Generate and save an inkblot to file"""
        fig = self.create_inkblot(use_color, complexity, bizarre_mode)
        plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1,
                    facecolor='#f8f9fa', edgecolor='none')
        plt.close(fig)
        print(f"Inkblot saved as {filename}")

    def show_inkblot(self, use_color=False, complexity=3, bizarre_mode=False):
        """Generate and display an inkblot"""
        fig = self.create_inkblot(use_color, complexity, bizarre_mode)
        plt.show()
        plt.close(fig)


# Example usage and demonstration
def main():
    # Create generator instance
    generator = InkblotGenerator()

    print("Python Inkblot Generator")
    print("=" * 40)

    # Generate different examples
    examples = [
        {"name": "Classic B&W Simple", "color": False, "complexity": 1,
         "bizarre": False},
        {"name": "Classic B&W Complex", "color": False, "complexity": 5,
         "bizarre": False},
        {"name": "Colored Medium", "color": True, "complexity": 3,
         "bizarre": False},
        {"name": "Bizarre Colored", "color": True, "complexity": 4,
         "bizarre": True},
    ]

    for i, example in enumerate(examples):
        print(f"\nGenerating {example['name']}...")
        filename = f"inkblot_{i + 1}_{example['name'].lower().replace(' ', '_')}.png"
        generator.save_inkblot(
            filename,
            use_color=example['color'],
            complexity=example['complexity'],
            bizarre_mode=example['bizarre']
        )

    print(f"\n✅ Generated {len(examples)} example inkblots!")
    print("\nTo create custom inkblots:")
    print("generator = InkblotGenerator()")
    print(
        "generator.save_inkblot('my_inkblot.png', use_color=True, complexity=3, bizarre_mode=False)")
    print("# or")
    print(
        "generator.show_inkblot(use_color=True, complexity=3, bizarre_mode=False)")


# Interactive function for easy use
def generate_custom_inkblot():
    """Interactive function to generate custom inkblots"""
    generator = InkblotGenerator()

    print("\n🎨 Custom Inkblot Generator")
    print("-" * 30)

    # Get user preferences
    use_color = input("Use colors? (y/n): ").lower().startswith('y')

    try:
        complexity = int(input("Complexity level (1-5): "))
        complexity = max(1, min(5, complexity))  # Clamp to valid range
    except ValueError:
        complexity = 3
        print("Invalid input, using complexity 3")

    bizarre = input("Bizarre mode (asymmetric)? (y/n): ").lower().startswith(
        'y')

    filename = input("Output filename (or press Enter for default): ").strip()
    if not filename:
        filename = f"custom_inkblot_c{complexity}_{'color' if use_color else 'bw'}_{'bizarre' if bizarre else 'normal'}.png"

    # Generate and save
    generator.save_inkblot(filename, use_color, complexity, bizarre)
    print(f"\n🎉 Your custom inkblot has been created: {filename}")


if __name__ == "__main__":
    main()

    # Uncomment the line below for interactive mode
    generate_custom_inkblot()