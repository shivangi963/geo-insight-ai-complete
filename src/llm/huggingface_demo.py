from transformers import pipeline

def sentiment_analysis_example():
    """Example of using Hugging Face for sentiment analysis"""
    classifier = pipeline("sentiment-analysis")
    
    # Test with multiple property-related phrases
    test_phrases = [
        "I love this beautiful apartment with amazing views!",
        "The neighborhood seems noisy and crowded.",
        "This property has great potential but needs some updates.",
        "Perfect location with excellent amenities and modern finishes."
    ]
    
    for text in test_phrases:
        result = classifier(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.3f}")
        print("-" * 50)
    
    return [classifier(text) for text in test_phrases]

def summarization_example():
    summarizer = pipeline("summarization")
    
    long_text = """ This stunning luxury apartment features three spacious bedrooms, two modern bathrooms, 
    and a state-of-the-art kitchen with stainless steel appliances. The living area boasts 
    floor-to-ceiling windows with panoramic city views, hardwood flooring throughout, 
    and a cozy fireplace. Located in the heart of downtown, it's just steps away from 
    restaurants, shopping centers, and public transportation. The building offers 
    amenities including a 24-hour concierge, fitness center, swimming pool, and 
    secure underground parking. Perfect for professionals or small families seeking 
    luxury urban living with all modern conveniences."""
    
    summary = summarizer(
        long_text, 
        max_length=100, 
        min_length=30, 
        do_sample=False,
        truncation=True
    )
    
    print(f"Original text length: {len(long_text)} characters")
    print(f"Summary: {summary[0]['summary_text']}")
    return summary

if __name__ == "__main__":
    print("=== Hugging Face Demos ===")
    
    print("\n1. Sentiment Analysis:")
    sentiment_analysis_example()
    
    print("\n2. Text Summarization:")
    summarization_example()