import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gpt4_model():
    """Test the new GPT-4.1-2025-04-14 model"""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test prompt
    test_prompt = """
    You are a helpful assistant specializing in Manim (Mathematical Animation Engine). 
    Please briefly explain what Manim is and give one simple example of what you can create with it.
    """
    
    try:
        print("🧪 Testing GPT-4.1-2025-04-14 model...")
        print("=" * 50)
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",  # The new model
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in mathematical animations and Manim."},
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        # Extract and display response
        answer = response.choices[0].message.content
        
        print("✅ SUCCESS! Model is working correctly.")
        print("\n🤖 Model Response:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        
        # Display usage info
        usage = response.usage
        print(f"\n📊 Usage Statistics:")
        print(f"   • Prompt tokens: {usage.prompt_tokens}")
        print(f"   • Completion tokens: {usage.completion_tokens}")
        print(f"   • Total tokens: {usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to test model - {str(e)}")
        
        # Check if it's a model availability issue
        if "model" in str(e).lower():
            print("\n💡 Possible solutions:")
            print("   1. The model name might be incorrect")
            print("   2. The model might not be available in your region yet")
            print("   3. Your API key might not have access to this model")
            print("   4. Try using 'gpt-4o' or 'gpt-4-turbo' as alternatives")
        
        return False

if __name__ == "__main__":
    test_gpt4_model() 