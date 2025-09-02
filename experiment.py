"""
Main Experimental Runner for AI Rorschach Pilot Study
Coordinates the execution of personality-based image interpretations
"""

import os
import json
import random
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import time

from portal import analyze_image_with_ai
from personas import PersonaDefinitions
from generator import InkblotGenerator


class RorschachExperiment:
    """Main experimental runner for AI Rorschach study"""
    
    def __init__(self, output_dir="results", models=None):
        """
        Initialize the experiment
        
        Args:
            output_dir: Directory to save results
            models: List of models to use (default: ["gpt-4o"])
        """
        self.output_dir = Path(output_dir)
        self.models = models or ["gpt-4o"]
        self.personas = PersonaDefinitions.get_all_personas()
        
        # Create output directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories"""
        (self.output_dir / "responses").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "generated_inkblots").mkdir(parents=True, exist_ok=True)
        
    def select_images(self, n_per_category=5):
        """
        Select a diverse subset of images for the experiment
        
        Args:
            n_per_category: Number of images to select from each category
            
        Returns:
            List of image paths
        """
        selected_images = []
        
        # Select fractals
        fractal_dir = Path("raw_image/Fractal/Fractals_400_380x380_grey_SpecHist_SSIM")
        if fractal_dir.exists():
            fractal_files = [f for f in fractal_dir.glob("*.bmp") 
                           if not f.name.startswith("Thumbs")]
            if fractal_files:
                # Sample diverse fractals (different number ranges for variety)
                sampled_fractals = []
                ranges = [(1, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000)]
                for range_start, range_end in ranges:
                    range_files = [f for f in fractal_files 
                                 if range_start <= int(f.stem.split('_')[-1]) <= range_end]
                    if range_files:
                        sampled_fractals.append(random.choice(range_files))
                
                selected_images.extend(sampled_fractals[:n_per_category])
                print(f"Selected {len(sampled_fractals[:n_per_category])} fractal images")
        
        # Select art.pics images
        art_dir = Path("raw_image/art_pics/art.pics stimuli")
        if art_dir.exists():
            # Select one from each style
            styles = ["azulejos", "klimt", "munch", "pointilism"]
            art_samples = []
            for style in styles:
                style_files = list(art_dir.glob(f"*_{style}.jpg"))
                if style_files:
                    # Sample from different number ranges
                    sampled = random.sample(style_files, 
                                          min(2, len(style_files)))
                    art_samples.extend(sampled)
            
            selected_images.extend(art_samples[:n_per_category])
            print(f"Selected {len(art_samples[:n_per_category])} art.pics images")
        
        # Generate new inkblots
        print(f"Generating {n_per_category} new inkblot images...")
        inkblot_gen = InkblotGenerator()
        for i in range(n_per_category):
            complexity = random.randint(1, 5)
            use_color = random.choice([True, False])
            filename = self.output_dir / "generated_inkblots" / f"inkblot_{i+1}_c{complexity}_{'color' if use_color else 'bw'}.png"
            inkblot_gen.save_inkblot(str(filename), use_color=use_color, complexity=complexity)
            selected_images.append(filename)
        
        print(f"\nTotal images selected: {len(selected_images)}")
        return selected_images
    
    def run_interpretation(self, image_path, persona_trait, model="gpt-4o", 
                         temperature=1.0, num_iterations=3):
        """
        Run interpretation for a single image with a specific persona
        
        Args:
            image_path: Path to the image
            persona_trait: Big Five trait name
            model: Model to use
            temperature: Temperature for generation
            num_iterations: Number of interpretations to generate
            
        Returns:
            List of interpretation responses
        """
        responses = []
        prompt = PersonaDefinitions.get_interpretation_prompt(persona_trait)
        
        for iteration in range(num_iterations):
            try:
                # Add slight delay to avoid rate limiting
                if iteration > 0:
                    time.sleep(2)
                
                # Get interpretation
                response = analyze_image_with_ai(
                    str(image_path),
                    model=model,
                    prompt=prompt,
                    max_size_mb=3
                )
                
                # Format with metadata
                formatted_response = PersonaDefinitions.format_response_metadata(
                    trait_name=persona_trait,
                    image_id=Path(image_path).stem,
                    model=model,
                    response=response,
                    temperature=temperature,
                    seed=iteration
                )
                
                responses.append(formatted_response)
                print(f"  ✓ Iteration {iteration + 1}/{num_iterations} complete")
                
            except Exception as e:
                print(f"  ✗ Error in iteration {iteration + 1}: {e}")
                continue
        
        return responses
    
    def run_full_experiment(self, n_images_per_category=3, n_iterations=3, 
                          temperature=0.9, personas_to_test=None):
        """
        Run the full experiment across all personas and images
        
        Args:
            n_images_per_category: Number of images per category
            n_iterations: Number of iterations per persona/image combo
            temperature: Temperature for generation
            personas_to_test: List of persona traits to test (default: all)
            
        Returns:
            Path to results summary file
        """
        print("\n" + "=" * 60)
        print("Starting AI Rorschach Experiment")
        print("=" * 60)
        
        # Select images
        print("\n1. Selecting Images...")
        images = self.select_images(n_images_per_category)
        
        # Determine which personas to test
        if personas_to_test is None:
            personas_to_test = list(self.personas.keys())
        
        all_results = []
        experiment_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "n_images": len(images),
            "n_personas": len(personas_to_test),
            "n_iterations": n_iterations,
            "temperature": temperature,
            "models": self.models,
            "personas_tested": personas_to_test
        }
        
        # Run interpretations
        print(f"\n2. Running Interpretations...")
        print(f"   - {len(personas_to_test)} personas")
        print(f"   - {len(images)} images")
        print(f"   - {n_iterations} iterations each")
        print(f"   - Total: {len(personas_to_test) * len(images) * n_iterations} interpretations\n")
        
        for model in self.models:
            print(f"\nUsing model: {model}")
            print("-" * 40)
            
            for persona_trait in personas_to_test:
                persona_info = self.personas[persona_trait]
                print(f"\nPersona: {persona_info['name']} ({persona_trait})")
                
                for img_idx, image_path in enumerate(images):
                    print(f" Image {img_idx + 1}/{len(images)}: {Path(image_path).name}")
                    
                    responses = self.run_interpretation(
                        image_path=image_path,
                        persona_trait=persona_trait,
                        model=model,
                        temperature=temperature,
                        num_iterations=n_iterations
                    )
                    
                    # Save individual responses
                    for response in responses:
                        all_results.append(response)
                        
                        # Save to file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = (f"{model}_{persona_trait}_{Path(image_path).stem}_"
                                  f"{response['generation_params']['seed']}_{timestamp}.json")
                        filepath = self.output_dir / "responses" / filename
                        
                        with open(filepath, 'w') as f:
                            json.dump(response, f, indent=2)
        
        # Save experiment summary
        summary = {
            "metadata": experiment_metadata,
            "images_used": [str(img) for img in images],
            "total_responses": len(all_results),
            "responses_by_persona": {
                trait: len([r for r in all_results if r["persona"]["trait"] == self.personas[trait]["trait"]])
                for trait in personas_to_test
            }
        }
        
        summary_path = self.output_dir / f"experiment_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Experiment Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary: {summary_path}")
        print("=" * 60)
        
        return summary_path
    
    def run_quick_demo(self, test_image_path=None):
        """
        Run a quick demonstration with one image and all personas
        
        Args:
            test_image_path: Optional path to test image
            
        Returns:
            Dict of responses by persona
        """
        print("\n" + "=" * 60)
        print("AI Rorschach Quick Demo")
        print("=" * 60)
        
        # Use provided image or generate a new inkblot
        if test_image_path and Path(test_image_path).exists():
            image_path = test_image_path
            print(f"\nUsing provided image: {image_path}")
        else:
            print("\nGenerating test inkblot...")
            inkblot_gen = InkblotGenerator()
            image_path = self.output_dir / "generated_inkblots" / "demo_inkblot.png"
            inkblot_gen.save_inkblot(str(image_path), use_color=True, complexity=3)
        
        demo_results = {}
        
        print(f"\nRunning interpretations with {len(self.personas)} personas...")
        print("-" * 40)
        
        for trait_name, persona_info in self.personas.items():
            print(f"\n{persona_info['name']} ({trait_name}):")
            
            try:
                prompt = PersonaDefinitions.get_interpretation_prompt(trait_name)
                response = analyze_image_with_ai(
                    str(image_path),
                    model=self.models[0],
                    prompt=prompt
                )
                
                demo_results[trait_name] = {
                    "persona": persona_info['name'],
                    "style": persona_info['interpretation_style'],
                    "interpretation": response
                }
                
                # Display truncated response
                print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
                
            except Exception as e:
                print(f"  Error: {e}")
                demo_results[trait_name] = {"error": str(e)}
        
        # Save demo results
        demo_path = self.output_dir / f"demo_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(demo_path, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"Demo complete! Results saved to: {demo_path}")
        print("=" * 60)
        
        return demo_results


def main():
    """Main entry point for running experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Rorschach Experiment Runner")
    parser.add_argument("--mode", choices=["demo", "pilot", "full"], default="demo",
                       help="Experiment mode to run")
    parser.add_argument("--models", nargs="+", default=["gpt-4o"],
                       help="Models to use (e.g., gpt-4o gemini claude)")
    parser.add_argument("--n-images", type=int, default=3,
                       help="Number of images per category")
    parser.add_argument("--n-iterations", type=int, default=3,
                       help="Number of iterations per persona/image")
    parser.add_argument("--temperature", type=float, default=0.9,
                       help="Temperature for generation")
    parser.add_argument("--test-image", type=str,
                       help="Path to specific test image for demo mode")
    parser.add_argument("--personas", nargs="+",
                       choices=["openness", "conscientiousness", "extraversion", 
                               "agreeableness", "neuroticism"],
                       help="Specific personas to test (default: all)")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = RorschachExperiment(models=args.models)
    
    if args.mode == "demo":
        # Run quick demo
        experiment.run_quick_demo(test_image_path=args.test_image)
        
    elif args.mode == "pilot":
        # Run pilot with fewer images/iterations
        experiment.run_full_experiment(
            n_images_per_category=2,
            n_iterations=2,
            temperature=args.temperature,
            personas_to_test=args.personas
        )
        
    elif args.mode == "full":
        # Run full experiment
        experiment.run_full_experiment(
            n_images_per_category=args.n_images,
            n_iterations=args.n_iterations,
            temperature=args.temperature,
            personas_to_test=args.personas
        )


if __name__ == "__main__":
    main()