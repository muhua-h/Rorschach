"""
Enhanced AI Rorschach Experiment - Interactive HTML with High/Low Personas
"""

import os
import random
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import io
import time
import json
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
        persona_trait: Persona trait name (with _high or _low suffix)
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


def process_image_parallel(image_path: Path, trait_personas: Dict[str, list], 
                          model: str, temperature: float, 
                          max_workers: int = 10) -> Dict[str, Any]:
    """
    Process all personas (high and low) for a single image in parallel
    
    Args:
        image_path: Path to image
        trait_personas: Dictionary of traits with high/low persona lists
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
    
    # Flatten all personas to process
    all_personas = []
    for trait_name, personas in trait_personas.items():
        for persona in personas:
            # Create key like "openness_high" or "openness_low"
            level = "high" if "High" in persona["trait"] else "low"
            key = f"{trait_name}_{level}"
            all_personas.append((key, persona))
    
    # Process personas in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                analyze_with_retry, 
                image_path, 
                persona_key, 
                model, 
                temperature
            ): (persona_key, persona_info)
            for persona_key, persona_info in all_personas
        }
        
        # Collect results as they complete
        completed = 0
        total = len(all_personas)
        
        for future in as_completed(futures):
            persona_key, persona_info = futures[future]
            completed += 1
            
            try:
                key, response = future.result()
                row_data["interpretations"][key] = response
                
                if "Failed" in response or "Error" in response:
                    print(f"  [{completed}/{total}] {persona_info['name']}: ✗")
                else:
                    print(f"  [{completed}/{total}] {persona_info['name']}: ✓")
                    
            except Exception as e:
                row_data["interpretations"][persona_key] = f"Unexpected error: {str(e)}"
                print(f"  [{completed}/{total}] {persona_info['name']}: ✗ (Exception)")
    
    return row_data


def run_enhanced_experiment(model="gpt-4o", temperature=0.9, use_parallel=True, max_workers=10):
    """
    Run enhanced experiment with high/low personas and generate interactive HTML
    
    Args:
        model: Model to use for analysis
        temperature: Temperature for generation
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
    """
    
    print("\n" + "="*60)
    print("AI RORSCHACH ENHANCED EXPERIMENT")
    print("="*60)
    
    # Get images
    print("\n1. Selecting images...")
    images = select_diverse_images()
    
    # Get personas organized by trait (high and low)
    trait_personas = PersonaDefinitions.get_trait_personas(include_low=True)
    
    # Count total personas
    total_personas = sum(len(personas) for personas in trait_personas.values())
    
    print(f"\n2. Running interpretations...")
    print(f"   - {len(images)} images")
    print(f"   - {total_personas} personas ({len(trait_personas)} traits × 2 levels)")
    print(f"   - Total: {len(images) * total_personas} interpretations")
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
                trait_personas, 
                model, 
                temperature,
                max_workers
            )
            results.append(row_data)
    else:
        # Sequential processing with retry
        for img_idx, image_path in enumerate(images):
            img_name = Path(image_path).stem
            print(f"\n[Image {img_idx+1}/{len(images)}] {img_name}")
            
            row_data = {
                "image_path": str(image_path),
                "image_name": img_name,
                "image_base64": image_to_base64(image_path),
                "interpretations": {}
            }
            
            for trait_name, personas in trait_personas.items():
                for persona in personas:
                    level = "high" if "High" in persona["trait"] else "low"
                    key = f"{trait_name}_{level}"
                    
                    print(f"  - {persona['name']}...", end=" ")
                    
                    # Use retry logic
                    _, response = analyze_with_retry(
                        image_path,
                        key,
                        model,
                        temperature
                    )
                    row_data["interpretations"][key] = response
                    
                    if "Failed" in response or "Error" in response:
                        print("✗")
                    else:
                        print("✓")
            
            results.append(row_data)
    
    elapsed_time = time.time() - start_time
    
    # Generate HTML
    print("\n3. Generating interactive HTML report...")
    html = generate_interactive_html(results, trait_personas)
    
    # Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rorschach_enhanced_{timestamp}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n" + "="*60)
    print(f"✓ Experiment completed in {elapsed_time:.1f} seconds")
    print(f"✓ Results saved to: {output_file}")
    print("\nOpen this file in your browser to view the interactive results.")
    print("Features:")
    print("  - Hover over cells to see persona prompts")
    print("  - Click cells to toggle between high/low trait expressions")
    print("="*60)
    
    return output_file


def generate_interactive_html(results, trait_personas):
    """Generate interactive HTML with hover and click effects"""
    
    # Prepare persona prompts for JavaScript
    persona_prompts = {}
    for trait_name, personas in trait_personas.items():
        for persona in personas:
            level = "high" if "High" in persona["trait"] else "low"
            key = f"{trait_name}_{level}"
            # Escape the prompt for JavaScript
            prompt = PersonaDefinitions.get_interpretation_prompt(key, include_instructions=False)
            persona_prompts[key] = {
                "name": persona["name"],
                "trait": persona["trait"],
                "style": persona["interpretation_style"],
                "prompt": prompt.replace('\n', '\\n').replace('"', '\\"')
            }
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Rorschach Enhanced Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .instructions {{
            text-align: center;
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .results-table th {{
            background: #4a5568;
            color: white;
            padding: 15px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .results-table td {{
            padding: 15px;
            border: 1px solid #e2e8f0;
            vertical-align: top;
        }}
        .image-cell {{
            text-align: center;
            background: #f8f9fa;
            min-width: 220px;
        }}
        .image-cell img {{
            max-width: 200px;
            max-height: 200px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }}
        .image-name {{
            font-size: 12px;
            color: #666;
            word-break: break-all;
        }}
        .interpretation-cell {{
            max-width: 350px;
            font-size: 14px;
            line-height: 1.6;
            cursor: pointer;
            position: relative;
            transition: background-color 0.3s;
        }}
        .interpretation-cell:hover {{
            background-color: #f0f8ff;
        }}
        .interpretation-cell.showing-low {{
            background-color: #fff5f5;
        }}
        .small-image {{
            float: right;
            max-width: 60px;
            max-height: 60px;
            margin-left: 10px;
            margin-bottom: 5px;
            border: 1px solid #ddd;
            opacity: 0.8;
        }}
        .persona-header {{
            font-size: 14px;
            font-weight: 600;
            cursor: help;
        }}
        .persona-trait {{
            font-size: 11px;
            color: #888;
            font-weight: normal;
        }}
        .trait-toggle {{
            font-size: 10px;
            color: #007bff;
            cursor: pointer;
            text-decoration: underline;
        }}
        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        .metadata {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 14px;
        }}
        .tooltip {{
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -200px;
            width: 400px;
            font-size: 12px;
            line-height: 1.4;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .interpretation-cell:hover .tooltip {{
            visibility: visible;
        }}
        .prompt-preview {{
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: monospace;
            font-size: 11px;
            margin-top: 5px;
            padding: 5px;
            background: #444;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>AI Rorschach Enhanced Study Results</h1>
    <p class="subtitle">Interactive Personality-Based Interpretations with High/Low Trait Expressions</p>
    
    <div class="instructions">
        <strong>Interactive Features:</strong><br>
        🔍 <strong>Hover</strong> over interpretation cells to see the persona prompt<br>
        🔄 <strong>Click</strong> interpretation cells to toggle between high/low trait expressions
    </div>
    
    <table class="results-table">
        <thead>
            <tr>
                <th>Image</th>
"""
    
    # Add trait headers (initially showing high traits)
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    for trait in traits:
        high_persona = trait_personas[trait][0]
        html += f"""
                <th class="persona-header" id="header-{trait}">
                    <span class="persona-name">{high_persona['name']}</span><br>
                    <span class="persona-trait">({high_persona['trait']})</span><br>
                    <span class="trait-toggle" onclick="toggleAllTrait('{trait}')">⇄ Toggle All</span>
                </th>
"""
    
    html += """
            </tr>
        </thead>
        <tbody>
"""
    
    # Add data rows
    for row_idx, result in enumerate(results):
        html += f"""
            <tr id="row-{row_idx}">
                <td class="image-cell">
                    <img src="{result['image_base64']}" alt="{result['image_name']}">
                    <div class="image-name">{result['image_name']}</div>
                </td>
"""
        
        # Add interpretations for each trait
        for trait in traits:
            high_key = f"{trait}_high"
            low_key = f"{trait}_low"
            
            high_interp = result["interpretations"].get(high_key, "No interpretation")
            low_interp = result["interpretations"].get(low_key, "No interpretation")
            
            # Truncate if too long
            if len(high_interp) > 400:
                high_interp = high_interp[:397] + "..."
            if len(low_interp) > 400:
                low_interp = low_interp[:397] + "..."
            
            html += f"""
                <td class="interpretation-cell" 
                    data-trait="{trait}"
                    data-row="{row_idx}"
                    data-high="{high_interp.replace('"', '&quot;')}"
                    data-low="{low_interp.replace('"', '&quot;')}"
                    data-showing="high"
                    onclick="toggleCell(this)">
                    <div class="interpretation-text">{high_interp}</div>
                    <div class="tooltip">
                        <strong class="tooltip-persona-name">{persona_prompts[high_key]['name']}</strong><br>
                        <em>Style: {persona_prompts[high_key]['style']}</em>
                        <div class="prompt-preview">{persona_prompts[high_key]['prompt'][:300]}...</div>
                    </div>
                </td>
"""
        
        html += """            </tr>
"""
    
    html += f"""
        </tbody>
    </table>
    
    <div class="metadata">
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
        Images: {len(results)} | Traits: {len(traits)} | Levels: 2 (High/Low) | Total Interpretations: {len(results) * len(traits) * 2}
    </div>
    
    <script>
        // Persona prompts data
        const personaPrompts = {json.dumps(persona_prompts)};
        
        // Track which traits are showing high/low
        const traitStates = {{
            'openness': 'high',
            'conscientiousness': 'high',
            'extraversion': 'high',
            'agreeableness': 'high',
            'neuroticism': 'high'
        }};
        
        function toggleCell(cell) {{
            const trait = cell.dataset.trait;
            const currentShowing = cell.dataset.showing;
            const newShowing = currentShowing === 'high' ? 'low' : 'high';
            
            // Update cell content
            const newText = cell.dataset[newShowing];
            cell.querySelector('.interpretation-text').innerHTML = newText;
            cell.dataset.showing = newShowing;
            
            // Update cell styling
            if (newShowing === 'low') {{
                cell.classList.add('showing-low');
            }} else {{
                cell.classList.remove('showing-low');
            }}
            
            // Update tooltip
            const personaKey = trait + '_' + newShowing;
            const persona = personaPrompts[personaKey];
            cell.querySelector('.tooltip-persona-name').textContent = persona.name;
            cell.querySelector('.tooltip em').textContent = 'Style: ' + persona.style;
            cell.querySelector('.prompt-preview').textContent = persona.prompt.substring(0, 300) + '...';
        }}
        
        function toggleAllTrait(trait) {{
            // Toggle the trait state
            const currentState = traitStates[trait];
            const newState = currentState === 'high' ? 'low' : 'high';
            traitStates[trait] = newState;
            
            // Update all cells for this trait
            const cells = document.querySelectorAll(`[data-trait="${{trait}}"]`);
            cells.forEach(cell => {{
                if (cell.dataset.showing !== newState) {{
                    toggleCell(cell);
                }}
            }});
            
            // Update header
            const header = document.getElementById('header-' + trait);
            const personaKey = trait + '_' + newState;
            const persona = personaPrompts[personaKey];
            header.querySelector('.persona-name').textContent = persona.name;
            header.querySelector('.persona-trait').textContent = '(' + persona.trait + ')';
        }}
    </script>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AI Rorschach Experiment")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (default: 0.9)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=10, help="Maximum parallel workers (default: 10)")
    
    args = parser.parse_args()
    
    run_enhanced_experiment(
        model=args.model, 
        temperature=args.temperature,
        use_parallel=not args.no_parallel,
        max_workers=args.workers
    )