import os
from transformers import pipeline
import google.generativeai as genai

class PropertyDescriptionSummarizer:
    def __init__(self, huggingface_model="facebook/bart-large-cnn"):
        self.huggingface_model = huggingface_model
        self.gemini_api_key = ""  
        
    def set_gemini_api_key(self, api_key):    
        self.gemini_api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
    
    def huggingface_summarize(self, property_description):
       
        try:
            clean_description = ' '.join(property_description.split())
        
            summarizer = pipeline("summarization", model=self.huggingface_model)

            word_count = len(clean_description.split())

            max_length = min(150, max(80, word_count // 2))
            min_length = min(40, max_length // 2)
        
            summary = summarizer(
                clean_description,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error with Hugging Face: {e}")
            return None
    
   
    def gemini_summarize(self, property_description):
    
        try:
            model = genai.GenerativeModel('gemini-2.5-flash') 
            
            prompt =  f"""Create a concise, one-paragraph summary of this property rental description.
            Focus on key features, location advantages, and main benefits for potential renters.

            Property Description: {property_description}

            Summary:"""

            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return None
    
    
    def compare_summaries(self, property_description):
        """Compare summaries from all available methods"""
        print("=" * 80)
        print("ORIGINAL PROPERTY DESCRIPTION:")
        print("=" * 80)
    
        clean_description = ' '.join(property_description.split())
        print(clean_description)
        print(f"\nOriginal length: {len(clean_description)} characters")
        print("=" * 80)
        
        summaries = {}
        
        # Hugging Face summary
        print("\n1. HUGGING FACE SUMMARY:")
        print("-" * 40)
        hf_summary = self.huggingface_summarize(property_description)
        if hf_summary:
            print(hf_summary)
            summaries['Hugging Face'] = hf_summary
            print(f"Summary length: {len(hf_summary)} characters")
        
        # Gemini summary 
        print("\n2. GEMINI API SUMMARY:")
        print("-" * 40)
        gemini_summary = self.gemini_summarize(property_description)
        if gemini_summary:
            print(gemini_summary)
            summaries['Gemini'] = gemini_summary
            print(f"Summary length: {len(gemini_summary)} characters")
        else:
            print("Gemini summary not available")
        
        return summaries
    
    

def main():
    
    property_description = """Luxury 3-bedroom penthouse in downtown with panoramic city views.
Features include hardwood floors, gourmet kitchen with stainless steel appliances,
master suite with walk-in closet, and private balcony. Building amenities:
24-hour concierge, fitness center, swimming pool, and secure parking.
Located steps from restaurants, shopping, and public transportation.
Perfect for professionals seeking urban luxury living."""
    

    summarizer = PropertyDescriptionSummarizer()
    

    print("PROPERTY DESCRIPTION SUMMARIZER")
    print("-" * 50)
    summaries = summarizer.compare_summaries(property_description)
    
    print("\n" + "-" * 50)
    print("RESULTS SUMMARY:")
    for method, summary in summaries.items():
        print(f"{method}: {len(summary)} characters")
    
    return summaries

if __name__ == "__main__":
    summaries = main()
    