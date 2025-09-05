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

### Personas (Big Five Personality Traits - BFI-2 Framework)
We define **ten distinct AI personas** using the BFI-2 (Big Five Inventory-2) framework, with both high and low expressions for each trait:

#### Openness to Experience
1. **High Openness** - "The Creative Explorer"
   - Imaginative, artistic, curious, abstract thinking
   - Sees patterns, metaphors, and novel connections
2. **Low Openness** - "The Practical Observer"
   - Practical, conventional, concrete thinking
   - Focuses on literal, straightforward interpretations

#### Conscientiousness
3. **High Conscientiousness** - "The Systematic Analyst"  
   - Organized, detail-oriented, methodical
   - Focuses on structure, symmetry, and categorization
4. **Low Conscientiousness** - "The Spontaneous Observer"
   - Disorganized, careless, spontaneous
   - Provides casual, impressionistic observations

#### Extraversion
5. **High Extraversion** - "The Social Interpreter"
   - Energetic, social, action-oriented
   - Sees movement, interaction, and social scenarios
6. **Low Extraversion** - "The Quiet Contemplator"
   - Reserved, quiet, solitary
   - Focuses on stillness, calm, and introspection

#### Agreeableness
7. **High Agreeableness** - "The Harmonious Observer"
   - Cooperative, trusting, empathetic
   - Interprets positive, peaceful, and collaborative themes
8. **Low Agreeableness** - "The Critical Analyst"
   - Competitive, skeptical, critical
   - Notices conflict, competition, and negative aspects

#### Neuroticism
9. **High Neuroticism** - "The Anxious Perceiver"
   - Emotionally reactive, threat-sensitive
   - Notices potential dangers, conflicts, or negative elements
10. **Low Neuroticism** - "The Calm Observer"
    - Emotionally stable, resilient, calm
    - Sees balanced, stable, and harmonious elements

Each persona is implemented using **BFI-2 item descriptions** that combine all relevant trait indicators into comprehensive personality prompts.

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
  
- **`personas.py`**: Big Five personality trait definitions using BFI-2 framework
  - 10 distinct personas (5 traits × high/low expressions)
  - BFI-2 item-based personality descriptions
  - Systematic interpretation instructions
  - Metadata formatting for responses
  
- **`simple_experiment.py`**: Streamlined experimental runner
  - HTML table output with all interpretations
  - Parallel processing (5x faster)
  - Automatic retry with exponential backoff
  - Image selection from all three datasets
  
- **`enhanced_experiment.py`**: Interactive experimental runner with high/low personas
  - Interactive HTML with hover effects showing persona prompts
  - Click to toggle between high/low trait expressions
  - Supports all 10 personas (high and low for each Big Five trait)
  - Parallel processing with up to 10 workers
  - Visual indicators for trait levels

### Advanced Modules (Future Development)
- **`experiment.py`**: Full experimental framework with detailed logging
- **`analysis.py`**: Statistical analysis and visualization tools

### Running the Experiment

#### Dynamic Web Interface (NEW)
```bash
# Start the Flask backend server
python app.py

# Access the interface at http://localhost:5000
```

**Features:**
- Upload custom images or generate inkblots in-browser
- Create and edit custom personas with Big Five traits
- Random persona generator for quick experiments
- Real-time API integration with GPT-4, Gemini, Claude
- Gallery with image history and deletion capability
- Integrated settings panel (model, temperature, max tokens)
- Both backend (Python) and client-side (JavaScript) inkblot generation

**Interface Components:**
- **`dynamic_rorschach.html`**: Full-featured web interface
  - Drag-and-drop image upload
  - In-browser inkblot generator
  - Persona management with random defaults
  - Real-time analysis with multiple AI models
  - Gallery for viewing previous analyses
  
- **`app.py`**: Flask backend server
  - RESTful API endpoints for analysis
  - Batch processing for multiple personas
  - Inkblot generation endpoint
  - Integration with existing portal.py models

#### Simple Experiment (5 high traits only)
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

#### Enhanced Experiment (10 personas with high/low traits)
```bash
# Run enhanced experiment with interactive features
python enhanced_experiment.py

# With custom settings
python enhanced_experiment.py --model gemini --temperature 0.8

# Adjust parallel workers (default 10)
python enhanced_experiment.py --workers 15
```

#### Output Format

**Simple Experiment** (`rorschach_results_[timestamp].html`):
- 9 images (3 fractals, 3 art.pics, 3 generated inkblots)
- 5 personality interpretations per image (high traits only)
- Visual table format for easy comparison
- Total of 45 interpretations in one view

**Enhanced Experiment** (`rorschach_enhanced_[timestamp].html`):
- 9 images (3 fractals, 3 art.pics, 3 generated inkblots)
- 10 personas (5 traits × 2 levels each)
- Interactive features:
  - Hover to see BFI-2 based persona prompts
  - Click cells to toggle between high/low trait expressions
  - Toggle all cells for a trait at once
- Total of 90 interpretations (accessible via toggling)

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
