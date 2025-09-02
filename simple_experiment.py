"""
Simple AI Rorschach Experiment - Generate HTML table with interpretations
"""

import os
import random
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any

from portal import analyze_image_with_ai
from personas import PersonaDefinitions
from generator import InkblotGenerator


def image_to_base64(image_path, max_width=200):
    """Convert image to base64 for HTML embedding"""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles BMP, RGBA, etc.)
        if img.mode not in ('RGB', 'L'):
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = background
            else:
                img = img.convert('RGB')
        
        # Resize if too large
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to bytes
        buffer = io.BytesIO()
        # Use JPEG for better compatibility and smaller size
        if img.mode == 'L':  # Grayscale
            img.save(buffer, format='PNG')
            mime_type = 'png'
        else:
            img.save(buffer, format='JPEG', quality=90)
            mime_type = 'jpeg'
        buffer.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/{mime_type};base64,{img_str}"


def select_diverse_images():
    """Select 3 diverse images from each dataset"""
    selected = []
    
    # 1. Select 3 fractals (from different number ranges for diversity)
    fractal_dir = Path("raw_image/Fractal/Fractals_400_380x380_grey_SpecHist_SSIM")
    if fractal_dir.exists():
        fractal_files = [f for f in fractal_dir.glob("*.bmp") 
                        if not f.name.startswith("Thumbs")]
        
        # Pick from different ranges
        ranges = [(1, 200), (500, 1000), (1500, 1900)]
        for start, end in ranges:
            range_files = [f for f in fractal_files 
                          if start <= int(f.stem.split('_')[-1]) <= end]
            if range_files:
                selected.append(random.choice(range_files))
    
    # 2. Select 3 art.pics (different styles)
    art_dir = Path("raw_image/art_pics/art.pics stimuli")
    if art_dir.exists():
        styles = ["klimt", "munch", "pointilism"]
        for style in styles:
            style_files = list(art_dir.glob(f"*_{style}.jpg"))[:20]  # Limit search
            if style_files:
                selected.append(random.choice(style_files))
    
    # 3. Generate 3 inkblots
    inkblot_dir = Path("generated_inkblots")
    inkblot_dir.mkdir(exist_ok=True)
    
    generator = InkblotGenerator()
    configs = [
        {"complexity": 2, "color": False},
        {"complexity": 3, "color": True},
        {"complexity": 4, "color": False}
    ]
    
    for i, config in enumerate(configs):
        filename = inkblot_dir / f"inkblot_{i+1}_c{config['complexity']}_{'color' if config['color'] else 'bw'}.png"
        generator.save_inkblot(str(filename), use_color=config['color'], complexity=config['complexity'])
        selected.append(filename)
    
    print(f"Selected {len(selected)} images total")
    return selected


def analyze_with_retry(image_path: str, persona_trait: str, model: str, 
                      temperature: float, max_retries: int = 3, 
                      retry_delay: float = 2.0) -> Tuple[str, str]:
    """
    Analyze image with retry logic
    
    Args:
        image_path: Path to image
        persona_trait: Persona trait name
        model: Model to use
        temperature: Temperature for generation
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple of (persona_trait, response_text)
    """
    prompt = PersonaDefinitions.get_interpretation_prompt(persona_trait)
    
    for attempt in range(max_retries):
        try:
            response = analyze_image_with_ai(
                str(image_path),
                model=model,
                prompt=prompt,
                max_size_mb=3
            )
            return (persona_trait, response)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"    Retry {attempt + 1}/{max_retries - 1} for {persona_trait} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return (persona_trait, f"Failed after {max_retries} attempts: {str(e)}")
    
    return (persona_trait, "Failed: Unknown error")


def process_image_parallel(image_path: Path, personas: Dict[str, Any], 
                          model: str, temperature: float, 
                          max_workers: int = 5) -> Dict[str, Any]:
    """
    Process all personas for a single image in parallel
    
    Args:
        image_path: Path to image
        personas: Dictionary of persona definitions
        model: Model to use
        temperature: Temperature for generation
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with image data and interpretations
    """
    img_name = image_path.stem
    print(f"\nProcessing: {img_name}")
    
    # Prepare result structure
    row_data = {
        "image_path": str(image_path),
        "image_name": img_name,
        "image_base64": image_to_base64(image_path),
        "interpretations": {}
    }
    
    # Process personas in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                analyze_with_retry, 
                image_path, 
                persona_trait, 
                model, 
                temperature
            ): persona_trait
            for persona_trait in personas.keys()
        }
        
        # Collect results as they complete
        completed = 0
        total = len(personas)
        
        for future in as_completed(futures):
            persona_trait = futures[future]
            completed += 1
            
            try:
                trait, response = future.result()
                row_data["interpretations"][trait] = response
                
                if "Failed" in response or "Error" in response:
                    print(f"  [{completed}/{total}] {personas[trait]['name']}: ✗")
                else:
                    print(f"  [{completed}/{total}] {personas[trait]['name']}: ✓")
                    
            except Exception as e:
                row_data["interpretations"][persona_trait] = f"Unexpected error: {str(e)}"
                print(f"  [{completed}/{total}] {personas[persona_trait]['name']}: ✗ (Exception)")
    
    return row_data


def run_simple_experiment(model="gpt-4o", temperature=0.9, use_parallel=True, max_workers=5):
    """
    Run experiment and generate HTML table
    
    Args:
        model: Model to use for analysis
        temperature: Temperature for generation
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
    """
    
    print("\n" + "="*60)
    print("AI RORSCHACH SIMPLE EXPERIMENT")
    print("="*60)
    
    # Get images
    print("\n1. Selecting images...")
    images = select_diverse_images()
    
    # Get personas
    personas = PersonaDefinitions.get_all_personas()
    
    print(f"\n2. Running interpretations...")
    print(f"   - {len(images)} images")
    print(f"   - {len(personas)} personas")
    print(f"   - Total: {len(images) * len(personas)} interpretations")
    print(f"   - Mode: {'Parallel' if use_parallel else 'Sequential'} processing")
    if use_parallel:
        print(f"   - Max workers: {max_workers}\n")
    else:
        print()
    
    start_time = time.time()
    results = []
    
    if use_parallel:
        # Process each image with parallel persona analysis
        for img_idx, image_path in enumerate(images):
            print(f"\n[Image {img_idx+1}/{len(images)}]", end="")
            row_data = process_image_parallel(
                image_path, 
                personas, 
                model, 
                temperature,
                max_workers
            )
            results.append(row_data)
    else:
        # Original sequential processing with retry
        for img_idx, image_path in enumerate(images):
            img_name = Path(image_path).stem
            print(f"\n[Image {img_idx+1}/{len(images)}] {img_name}")
            
            row_data = {
                "image_path": str(image_path),
                "image_name": img_name,
                "image_base64": image_to_base64(image_path),
                "interpretations": {}
            }
            
            for persona_trait in personas.keys():
                print(f"  - {personas[persona_trait]['name']}...", end=" ")
                
                # Use retry logic even in sequential mode
                trait, response = analyze_with_retry(
                    image_path,
                    persona_trait,
                    model,
                    temperature
                )
                row_data["interpretations"][trait] = response
                
                if "Failed" in response or "Error" in response:
                    print("✗")
                else:
                    print("✓")
            
            results.append(row_data)
    
    elapsed_time = time.time() - start_time
    
    # Generate HTML
    print("\n3. Generating HTML report...")
    html = generate_html_table(results, personas)
    
    # Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rorschach_results_{timestamp}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n" + "="*60)
    print(f"✓ Experiment completed in {elapsed_time:.1f} seconds")
    print(f"✓ Results saved to: {output_file}")
    print("\nOpen this file in your browser to view the results table.")
    print("="*60)
    
    return output_file


def generate_html_table(results, personas):
    """Generate HTML with results table"""
    
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Rorschach Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-table th {
            background: #4a5568;
            color: white;
            padding: 15px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .results-table td {
            padding: 15px;
            border: 1px solid #e2e8f0;
            vertical-align: top;
        }
        .image-cell {
            text-align: center;
            background: #f8f9fa;
            min-width: 220px;
        }
        .image-cell img {
            max-width: 200px;
            max-height: 200px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
        .image-name {
            font-size: 12px;
            color: #666;
            word-break: break-all;
        }
        .interpretation-cell {
            max-width: 300px;
            font-size: 14px;
            line-height: 1.6;
        }
        .persona-header {
            font-size: 14px;
            font-weight: 600;
        }
        .persona-trait {
            font-size: 11px;
            color: #888;
            font-weight: normal;
        }
        tr:nth-child(even) {
            background-color: #f9fafb;
        }
        .metadata {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>AI Rorschach Pilot Study Results</h1>
    <p class="subtitle">Personality-Based Interpretations of Ambiguous Images</p>
    
    <table class="results-table">
        <thead>
            <tr>
                <th>Image</th>
"""
    
    # Add persona headers
    for trait_name, persona_info in personas.items():
        html += f"""
                <th class="persona-header">
                    {persona_info['name']}<br>
                    <span class="persona-trait">({persona_info['trait']})</span>
                </th>
"""
    
    html += """
            </tr>
        </thead>
        <tbody>
"""
    
    # Add data rows
    for result in results:
        html += """
            <tr>
                <td class="image-cell">
"""
        html += f'                    <img src="{result["image_base64"]}" alt="{result["image_name"]}">\n'
        html += f'                    <div class="image-name">{result["image_name"]}</div>\n'
        html += """                </td>
"""
        
        # Add interpretations
        for trait_name in personas.keys():
            interpretation = result["interpretations"].get(trait_name, "No interpretation")
            # Truncate if too long
            if len(interpretation) > 500:
                interpretation = interpretation[:497] + "..."
            
            html += f"""                <td class="interpretation-cell">{interpretation}</td>
"""
        
        html += """            </tr>
"""
    
    html += f"""
        </tbody>
    </table>
    
    <div class="metadata">
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
        Images: {len(results)} | Personas: {len(personas)} | Total Interpretations: {len(results) * len(personas)}
    </div>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple AI Rorschach Experiment")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (default: 0.9)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=5, help="Maximum parallel workers (default: 5)")
    
    args = parser.parse_args()
    
    run_simple_experiment(
        model=args.model, 
        temperature=args.temperature,
        use_parallel=not args.no_parallel,
        max_workers=args.workers
    )