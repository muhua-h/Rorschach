# AI Rorschach Pilot Study  
*Exploring Personality-Specific Interpretations of Ambiguous Images*

## Overview
This project is a **pilot study** investigating how AI agents with different **Big Five personality profiles** interpret ambiguous visual stimuli. The central idea is to use **projective methods** (inkblot-like tasks) to probe how value- and trait-oriented priors shape interpretation.  

By comparing interpretations across different personas, we aim to demonstrate that:
- Ambiguous stimuli (fractals, abstract art, Rorschach-like images) reveal **projection patterns**.  
- These projection patterns vary systematically with **personality steering** (Big Five traits).  
- This approach lays the groundwork for **AI as comparative minds** in cultural psychology and sociology.

This repository contains the **data, code, and experimental setup** for running the pilot.

---

## Technical Infrastructure

### Available Models
Through `portal.py`, we have access to:
- **Vision Models**: GPT-4o, GPT-4o-mini, Gemini 2.0 Flash, Claude 3.7 Sonnet, Azure Computer Vision
- **Text Models**: GPT-4 variants, DeepSeek-V3, Llama-3.3-70B, Claude, Gemini
- **Multimodal Capabilities**: All vision models support combined image + text prompts for personality-steered interpretations

### Image Generation
The `generator.py` module provides:
- Symmetric inkblot generation with customizable complexity
- Color/B&W options
- Bizarre mode for asymmetric patterns
- Batch generation capabilities

---

## Datasets
We combine three sources of ambiguous imagery:
1. **Fractal Stimuli** – 400 greyscale fractals from Ovalle-Fresa et al. (2022) with human ratings (located in `raw_image/Fractal/`)
   - Currently have 400 SHINEd fractals (380x380px, greyscale)
   - Metadata available in `mata-data.json`
2. **art.pics** – Style-transferred mundane objects from Thieleking et al. (2020) (located in `raw_image/art_pics/`)
   - Multiple artistic styles: azulejos, klimt, munch, pointilism
   - Includes normative ratings in CSV metadata files
3. **Self-Generated Rorschach-Like Inkblots** – synthetic, symmetric inkblots created via `generator.py`

For the pilot, we sample a **small, diverse subset** (e.g. 15–20 images per source).

---

## Experimental Design

### Personas (Big Five Personality Traits)
We define **five distinct AI personas**, each emphasizing one dominant Big Five trait:

1. **Openness** (High) - "The Creative Explorer"
   - Imaginative, artistic, curious, abstract thinking
   - Sees patterns, metaphors, and novel connections
   
2. **Conscientiousness** (High) - "The Systematic Analyst"  
   - Organized, detail-oriented, methodical
   - Focuses on structure, symmetry, and categorization
   
3. **Extraversion** (High) - "The Social Interpreter"
   - Energetic, social, action-oriented
   - Sees movement, interaction, and social scenarios
   
4. **Agreeableness** (High) - "The Harmonious Observer"
   - Cooperative, trusting, empathetic
   - Interprets positive, peaceful, and collaborative themes
   
5. **Neuroticism** (High) - "The Anxious Perceiver"
   - Emotionally reactive, threat-sensitive
   - Notices potential dangers, conflicts, or negative elements

Each persona is implemented by **steering multimodal models** using prompt-based personality frames combined with systematic instruction sets.

### Procedure
1. **Input**: Each persona is shown the same ambiguous image.  
2. **Task**: Generate 3–5 free-text interpretations per image (different seeds/temperatures).  
3. **Output**: Responses are logged with persona label, image ID, and metadata (token count, confidence).  
4. **Analysis**:
   - **Qualitative**: Compare tone, content themes, metaphor use across personas.  
   - **Quantitative**: Content coding (animals, humans, objects, affect, valence); embedding similarity; diversity metrics.  
   - **Reliability**: Check consistency of persona responses across seeds.

---

## Implementation Components

### Core Modules (Completed)
- **`portal.py`**: Unified interface for all AI models (text and vision)
  - Supports GPT-4o, Gemini, Claude, DeepSeek, Llama
  - Handles image format conversion (BMP, JPEG, PNG)
  - Automatic resizing for API limits
  
- **`generator.py`**: Inkblot generation with customizable parameters
  - Symmetric/asymmetric patterns
  - Color/grayscale options
  - Complexity levels (1-5)
  
- **`personas.py`**: Big Five personality trait definitions
  - 5 distinct personas with detailed prompts
  - Systematic interpretation instructions
  - Metadata formatting for responses
  
- **`simple_experiment.py`**: Streamlined experimental runner
  - HTML table output with all interpretations
  - Parallel processing (5x faster)
  - Automatic retry with exponential backoff
  - Image selection from all three datasets

### Advanced Modules (Future Development)
- **`experiment.py`**: Full experimental framework with detailed logging
- **`analysis.py`**: Statistical analysis and visualization tools

### Running the Experiment

#### Quick Start
```bash
# Run with default settings (parallel, GPT-4o)
python simple_experiment.py

# Sequential processing
python simple_experiment.py --no-parallel

# Different model
python simple_experiment.py --model gemini

# Adjust parallelization
python simple_experiment.py --workers 3
```

#### Output Format
The experiment generates an HTML file (`rorschach_results_[timestamp].html`) containing:
- 9 images (3 fractals, 3 art.pics, 3 generated inkblots)
- 5 personality interpretations per image
- Visual table format for easy comparison
- Total of 45 interpretations in one view

### Key Features
- **Robustness**: Automatic retry logic handles API failures
- **Performance**: Parallel processing reduces runtime by ~80%
- **Flexibility**: Support for multiple vision models
- **Accessibility**: Simple HTML output viewable in any browser

---

## Vision & Next Steps
This pilot is the **proof-of-concept** for a broader research program:
- **Step 1 (Pilot)**: Show that Big Five–steered personas produce distinct projection patterns.  
- **Step 2 (Expansion)**: Scale up to full Big Five profiles, value-based personas, and cultural frames.  
- **Step 3 (Human Comparison)**: Collect human responses to the same images, compare projection tendencies.  
- **Step 4 (Acculturation)**: Place multiple personas in dialogue to see if meanings converge (AI acculturation).  

Ultimately, this work aims to **bridge AI × psychology × sociology** by:
- Introducing **AI Rorschach** as a tool for studying projection, values, and cultural frames.  
- Positioning AI as a **comparative mind** for cultural psychology research.  
- Highlighting implications for **AI alignment, governance, and cultural integration**.

---
