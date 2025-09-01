import os
from dotenv import load_dotenv
import requests
import json
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from google import genai
import anthropic
from google.genai import types
import base64
from PIL import Image
import io
# Load environment variables
load_dotenv()


def get_model_response(model, user_prompt,
                       system_prompt="You are a helpful assistant.",
                       temperature=1.0, max_tokens=2048):
    """
    Universal function to get responses from various AI models.

    Args:
        model (str): The model to use (e.g., "gpt-4", "claude", "llama", "deepseek", "gemini")
        user_prompt (str): The user's input prompt
        system_prompt (str): System instructions for the AI model
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate

    Returns:
        str: The model's response text
    """
    model = model.lower()

    # OpenAI models via Azure
    if model in ["gpt-4", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                 "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini"]:
        return _get_azure_openai_response(model, user_prompt, system_prompt,
                                          temperature, max_tokens)

    # DeepSeek via Azure AI Inference
    elif model == "deepseek":
        return _get_deepseek_response(user_prompt, system_prompt, temperature,
                                      max_tokens)

    # Llama via Fireworks.ai
    elif model == "llama":
        return _get_llama_response(user_prompt, system_prompt, temperature,
                                   max_tokens)

    # Claude via Anthropic
    elif model == "claude":
        return _get_claude_response(user_prompt, system_prompt, temperature,
                                    max_tokens)

    # Gemini via Google
    elif model == "gemini":
        return _get_gemini_response(user_prompt, system_prompt, temperature,
                                    max_tokens)

    else:
        raise ValueError(f"Unsupported model: {model}")


def _get_azure_openai_response(model_name, user_prompt, system_prompt,
                               temperature, max_tokens):
    """Get response from Azure-hosted OpenAI models"""
    endpoint = "https://allmodelapi3225011299.openai.azure.com/"
    azure_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-12-01-preview"  # Adjust API version as needed

    # Map generic model names to Azure deployments
    deployment_map = {
        "gpt-4": "gpt-4",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }

    deployment = deployment_map.get(model_name, model_name)

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=azure_key,
    )

    try:
        response = client.chat.completions.create(
            model=deployment,  # Specify model parameter first
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Azure OpenAI error: {e}")
        raise


def _get_deepseek_response(user_prompt, system_prompt, temperature, max_tokens):
    """Get response from DeepSeek model via Azure AI Inference"""
    azure_key = os.getenv("AZURE_API_KEY")
    endpoint = "https://allmodelapi3225011299.services.ai.azure.com/models"
    model_name = "DeepSeek-V3"

    try:
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(azure_key),

        )

        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            model=model_name
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek error: {e}")
        raise

def _get_llama_response(user_prompt, system_prompt, temperature, max_tokens):
    """ Get response from llama model via Azure AI Inference"""
    azure_key = os.getenv("AZURE_API_KEY")
    endpoint = "https://allmodelapi3225011299.services.ai.azure.com/models"
    model_name = "Llama-3.3-70B-Instruct"

    try:
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(azure_key),

        )

        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            model=model_name
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Llama error: {e}")
        raise






def _get_claude_response(user_prompt, system_prompt, temperature, max_tokens):
    """Get response from Anthropic's Claude model"""
    api_key=os.environ.get("CLAUDE_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing ANTHROPIC_API_KEY or CLAUDE_API_KEY in environment variables")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        raise


def _get_gemini_response(user_prompt, system_prompt, temperature=0.7, max_tokens=512):
    """Get response from Gemini using generate_content (single-turn)"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    try:
        client = genai.Client(api_key=api_key)

        # Combine system and user prompts into one Content object
        content = types.Content(
            role="user",
            parts=[
                types.Part(text=f"{system_prompt}\n\n{user_prompt}")
            ]
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[content],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.text

    except Exception as e:
        print(f"Gemini API error: {e}")
        raise


def analyze_image_azure_vision(image_path, features=None):
    """
    Analyze an image using Azure Computer Vision API
    
    Args:
        image_path (str): Path to the image file
        features (list): Visual features to extract (default: all available)
        
    Returns:
        dict: Analysis results from Azure Computer Vision
    """
    azure_key = os.getenv("AZURE_VISION_KEY", os.getenv("AZURE_API_KEY"))
    azure_endpoint = os.getenv("AZURE_VISION_ENDPOINT", "https://allmodelapi3225011299.cognitiveservices.azure.com/")
    
    if not azure_key or not azure_endpoint:
        raise ValueError("Azure Computer Vision credentials not found in environment variables")
    
    if features is None:
        features = [
            VisualFeatureTypes.categories,
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.objects,
            VisualFeatureTypes.faces,
            VisualFeatureTypes.color,
            VisualFeatureTypes.image_type,
            VisualFeatureTypes.adult
        ]
    
    try:
        client = ComputerVisionClient(
            azure_endpoint, 
            CognitiveServicesCredentials(azure_key)
        )
        
        # Check file size and resize if needed (Azure limit is 4MB)
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        if file_size_mb > 4:
            # Resize image to fit Azure's 4MB limit
            with Image.open(image_path) as img:
                resize_ratio = (3.5 / file_size_mb) ** 0.5
                new_width = int(img.width * resize_ratio)
                new_height = int(img.height * resize_ratio)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                img_byte_arr = io.BytesIO()
                # Convert RGBA to RGB if needed
                if img_resized.mode == 'RGBA':
                    rgb_img = Image.new('RGB', img_resized.size, (255, 255, 255))
                    rgb_img.paste(img_resized, mask=img_resized.split()[3])
                    rgb_img.save(img_byte_arr, format='JPEG', quality=85)
                else:
                    img_resized.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr.seek(0)
                
                analysis = client.analyze_image_in_stream(
                    img_byte_arr,
                    visual_features=features
                )
        else:
            with open(image_path, "rb") as image_stream:
                analysis = client.analyze_image_in_stream(
                    image_stream,
                    visual_features=features
                )
        
        result = {
            "description": analysis.description.captions[0].text if analysis.description.captions else "No description available",
            "confidence": analysis.description.captions[0].confidence if analysis.description.captions else 0,
            "tags": [{"name": tag.name, "confidence": tag.confidence} for tag in analysis.tags],
            "categories": [{"name": cat.name, "score": cat.score} for cat in analysis.categories],
            "objects": [{"object": obj.object_property, "confidence": obj.confidence, 
                        "rectangle": {"x": obj.rectangle.x, "y": obj.rectangle.y, 
                                    "w": obj.rectangle.w, "h": obj.rectangle.h}} 
                       for obj in analysis.objects] if hasattr(analysis, 'objects') else [],
            "faces": len(analysis.faces) if hasattr(analysis, 'faces') else 0,
            "dominant_colors": analysis.color.dominant_colors if hasattr(analysis, 'color') else [],
            "is_adult": analysis.adult.is_adult_content if hasattr(analysis, 'adult') else False,
            "adult_score": analysis.adult.adult_score if hasattr(analysis, 'adult') else 0
        }
        
        return result
        
    except Exception as e:
        print(f"Azure Computer Vision error: {e}")
        raise


def analyze_image_with_ai(image_path, model="gpt-4o", prompt=None, max_size_mb=5):
    """
    Analyze an image using AI models that support vision (GPT-4V, Gemini, Claude)
    
    Args:
        image_path (str): Path to the image file
        model (str): Model to use for analysis
        prompt (str): Custom prompt for image analysis
        max_size_mb (float): Maximum file size in MB (will resize if larger)
        
    Returns:
        str: AI model's analysis of the image
    """
    if prompt is None:
        prompt = "Please analyze this image and describe what you see in detail."
    
    # Check file size and resize if needed
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        # Resize image to fit size limit
        with Image.open(image_path) as img:
            # Calculate resize ratio
            resize_ratio = (max_size_mb / file_size_mb) ** 0.5
            new_width = int(img.width * resize_ratio)
            new_height = int(img.height * resize_ratio)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img_format = 'JPEG' if img.format in ['JPEG', 'JPG'] else 'PNG'
            img_resized.save(img_byte_arr, format=img_format, quality=85)
            img_byte_arr.seek(0)
            
            image_data = base64.b64encode(img_byte_arr.read()).decode()
    else:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
    
    model = model.lower()
    
    if model in ["gpt-4o", "gpt-4o-mini", "gpt-4-vision"]:
        return _analyze_image_openai(image_data, prompt, model)
    elif model == "gemini":
        return _analyze_image_gemini(image_path, prompt)
    elif model == "claude":
        return _analyze_image_claude(image_data, prompt)
    else:
        raise ValueError(f"Model {model} does not support image analysis")


def _analyze_image_openai(image_base64, prompt, model="gpt-4o"):
    """Analyze image using OpenAI GPT-4 Vision"""
    endpoint = "https://allmodelapi3225011299.openai.azure.com/"
    azure_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-12-01-preview"
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=azure_key,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Vision error: {e}")
        raise


def _analyze_image_gemini(image_path, prompt):
    """Analyze image using Google Gemini"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    try:
        client = genai.Client(api_key=api_key)
        
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
        
        content = types.Content(
            role="user",
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=img_byte_arr
                    )
                )
            ]
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[content]
        )
        return response.text
        
    except Exception as e:
        print(f"Gemini Vision error: {e}")
        raise


def _analyze_image_claude(image_base64, prompt):
    """Analyze image using Claude"""
    api_key = os.environ.get("CLAUDE_API_KEY")
    
    if not api_key:
        raise ValueError("Claude API key not found in environment variables")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude Vision error: {e}")
        raise
if __name__ == "__main__":
    import sys
    
    # Check if image test mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "test-image":
        print("Testing Image Recognition Portal\n" + "="*50)
        
        image_path = "img_1.png"
        
        # Test 1: Azure Computer Vision API (if available)
        print("\n1. Testing Azure Computer Vision API:")
        print("-" * 40)
        try:
            vision_result = analyze_image_azure_vision(image_path)
            print(f"Description: {vision_result['description']}")
            print(f"Confidence: {vision_result['confidence']:.2%}")
            print(f"Tags: {', '.join([tag['name'] for tag in vision_result['tags'][:5]])}")
            print(f"Objects detected: {len(vision_result['objects'])}")
            for obj in vision_result['objects'][:3]:
                print(f"  - {obj['object']} (confidence: {obj['confidence']:.2%})")
        except Exception as e:
            print(f"Azure Vision API not available or error: {e}")
            print("Note: Requires azure-cognitiveservices-vision-computervision package")
            print("Install with: pip install azure-cognitiveservices-vision-computervision")
        
        # Test 2: GPT-4 Vision
        print("\n2. Testing GPT-4 Vision:")
        print("-" * 40)
        try:
            gpt4_result = analyze_image_with_ai(
                image_path, 
                model="gpt-4o",
                prompt="Analyze this image and describe what you see. Include details about objects, colors, and composition."
            )
            print(f"GPT-4 Vision Analysis:\n{gpt4_result}")
        except Exception as e:
            print(f"GPT-4 Vision error: {e}")
        
        # Test 3: Gemini Vision
        print("\n3. Testing Gemini Vision:")
        print("-" * 40)
        try:
            gemini_result = analyze_image_with_ai(
                image_path,
                model="gemini",
                prompt="What do you see in this image? Describe the main elements and their arrangement."
            )
            print(f"Gemini Vision Analysis:\n{gemini_result}")
        except Exception as e:
            print(f"Gemini Vision error: {e}")
        
        # Test 4: Claude Vision (if API key available)
        print("\n4. Testing Claude Vision:")
        print("-" * 40)
        try:
            claude_result = analyze_image_with_ai(
                image_path,
                model="claude",
                prompt="Provide a detailed analysis of this image, including any text, objects, and visual elements.",
                max_size_mb=3  # Claude has stricter size limits
            )
            print(f"Claude Vision Analysis:\n{claude_result}")
        except Exception as e:
            print(f"Claude Vision error: {e}")
            print("Note: Requires CLAUDE_API_KEY in environment variables")
        
        print("\n" + "="*50)
        print("Image Recognition Portal Test Complete!")
        
    else:
        # Original text model tests
        print("Testing Text Generation Models\n" + "="*50)
        
        ## Test all models
        
        # deepseek
        response = get_model_response(
            model="deepseek",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
        print(f"deepseek response:\n{response}\n")
    
        # llama
        response = get_model_response(
            model="llama",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
    
        print(f"llama response:\n{response}\n")
    
        # claude
        # response = get_model_response(
        #     model="claude",
        #     user_prompt="What are three benefits of AI models in modern applications?",
        #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
        # )
        # print(f"claude response:\n{response}\n")
    
        # gemini
        response = get_model_response(
            model="gemini",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
        print(f"gemini response:\n{response}\n")
    
        # gpt-4
        response = get_model_response(
            model="gpt-4",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
        print(f"gpt-4 response:\n{response}\n")
    
        # gpt-4o
        response = get_model_response(
            model="gpt-4o",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
        print(f"gpt-4o response:\n{response}\n")
    
        # gpt-4.5-preview
        response = get_model_response(
            model="gpt-4.5-preview",
            user_prompt="What are three benefits of AI models in modern applications?",
            system_prompt="You are a technical expert who explains concepts clearly and concisely."
        )
        print(f"gpt-4.5-preview response:\n{response}\n")
        
        print("\nTo test image recognition, run: python portal.py test-image")
