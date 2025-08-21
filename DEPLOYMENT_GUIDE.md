import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import time

class OnlineSpeechToImage:
    def __init__(self):
        print("🎨 Loading Stable Diffusion model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimize
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            
        print("✅ Model loaded successfully!")
    
    def enhance_prompt(self, text):
        """Enhance prompt for better quality"""
        if not any(word in text.lower() for word in ['digital art', 'painting', 'photo']):
            text += ", digital art, high quality"
        if not any(word in text.lower() for word in ['detailed', '4k', 'masterpiece']):
            text += ", detailed, high quality"
        return text
    
    def generate_image(self, prompt, steps=20, guidance_scale=7.5):
        """Generate image from text prompt"""
        if not prompt.strip():
            return None, "Please enter a description"
        
        try:
            # Enhance prompt
            enhanced_prompt = self.enhance_prompt(prompt)
            
            print(f"🎨 Generating: {enhanced_prompt}")
            start_time = time.time()
            
            # Generate
            with torch.autocast(self.device):
                result = self.pipe(
                    enhanced_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=512,
                    height=512
                )
            
            generation_time = time.time() - start_time
            image = result.images[0]
            
            info = f"✅ Generated in {generation_time:.1f}s\n📝 Enhanced prompt: {enhanced_prompt}"
            
            return image, info
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

# Initialize generator
print("🚀 Starting Speech-to-Image Generator...")
generator = OnlineSpeechToImage()

# Create Gradio interface
def text_to_image(prompt, steps, guidance_scale):
    """Gradio function for text-to-image"""
    return generator.generate_image(prompt, int(steps), float(guidance_scale))

def audio_to_image(audio, steps, guidance_scale):
    """Gradio function for audio-to-image"""
    if audio is None:
        return None, "Please record or upload audio"
    
    try:
        import speech_recognition as sr
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Process audio
        with sr.AudioFile(audio) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        
        print(f"🎤 Transcribed: {text}")
        
        # Generate image
        image, info = generator.generate_image(text, int(steps), float(guidance_scale))
        info = f"🎤 Heard: {text}\n" + info
        
        return image, info
        
    except Exception as e:
        return None, f"❌ Audio processing error: {str(e)}"

# Create interface
with gr.Blocks(title="🎨 Speech-to-Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Speech-to-Image Generator
    
    Create beautiful images from your voice or text using open-source Stable Diffusion!
    
    **No API keys required** • **Completely free** • **High quality results**
    """)
    
    with gr.Tabs():
        # Text-to-Image Tab
        with gr.Tab("📝 Text to Image"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Image Description",
                        placeholder="Describe the image you want to create...",
                        lines=3
                    )
                    
                    with gr.Row():
                        text_steps = gr.Slider(10, 50, value=20, label="Quality Steps")
                        text_guidance = gr.Slider(5, 15, value=7.5, label="Guidance Scale")
                    
                    text_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                with gr.Column():
                    text_output = gr.Image(label="Generated Image")
                    text_info = gr.Textbox(label="Generation Info", lines=3)
            
            # Examples
            gr.Examples(
                examples=[
                    ["A beautiful sunset over mountains", 20, 7.5],
                    ["A cute cat sitting in a garden", 20, 7.5],
                    ["A futuristic city with flying cars", 25, 8.0],
                    ["An abstract painting with vibrant colors", 20, 7.5],
                ],
                inputs=[text_input, text_steps, text_guidance],
                outputs=[text_output, text_info],
                fn=text_to_image,
                cache_examples=False
            )
        
        # Audio-to-Image Tab
        with gr.Tab("🎤 Speech to Image"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Record or Upload Audio",
                        type="filepath"
                    )
                    
                    with gr.Row():
                        audio_steps = gr.Slider(10, 50, value=20, label="Quality Steps")
                        audio_guidance = gr.Slider(5, 15, value=7.5, label="Guidance Scale")
                    
                    audio_btn = gr.Button("🎨 Generate from Speech", variant="primary")
                
                with gr.Column():
                    audio_output = gr.Image(label="Generated Image")
                    audio_info = gr.Textbox(label="Generation Info", lines=4)
    
    # Connect functions
    text_btn.click(
        text_to_image,
        inputs=[text_input, text_steps, text_guidance],
        outputs=[text_output, text_info]
    )
    
    audio_btn.click(
        audio_to_image,
        inputs=[audio_input, audio_steps, audio_guidance],
        outputs=[audio_output, audio_info]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips for Better Results:
    - **Be specific**: "A red cat sitting on a wooden chair" vs "a cat"
    - **Add style**: "...in digital art style" or "...photorealistic"
    - **Quality words**: "high quality", "detailed", "4k"
    - **Increase steps**: 25-50 for higher quality (slower)
    
    ### 🎤 Speech Tips:
    - Speak clearly and slowly
    - Use simple, descriptive language
    - Avoid background noise
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Creates public link
    )