import os
import sys
from google import genai
from google.genai.errors import APIError

# --- Configuration ---
RECONSTRUCTION_MODEL = 'gemini-2.5-pro' # Strong model for complex, multi-step reasoning
GROUNDING_MODEL = 'gemini-2.5-flash'    # Good for fast, search-grounded tasks

# --- Multi-Function RECONSTRUCTION PROMPT ---
# This single prompt instructs Gemini to handle decryption, translation, 
# and slang reconstruction sequentially.
RECONSTRUCTION_PROMPT = """
You are Project Chronos, an AI Archeologist specialized in deciphering ancient digital slang and cryptic codes.
Your task is to perform the following steps sequentially on the provided 'Fragment':

1.  **Decryption:** If the 'Fragment' is Morse code (using '.', '-', and '/'), fully decrypt it into plain English text.
2.  **Translation:** If the resulting text from step 1 is not standard English, translate it into standard English.
3.  **Reconstruction:** Translate all slang, abbreviations, and cultural references in the text into modern, plain English, filling in any missing context to make the sentence whole and coherent.

Do not add any preamble, explanation, or notes. **Only provide the complete, final, reconstructed sentence.**

Fragment: "{}"
Reconstruction: 
"""
# --- End of REVISED PROMPT ---


# --- Input Handling (Simplified) ---

# All pre-processing logic is removed from Python; the raw fragment is sent to Gemini.
def get_fragment_input():
    """Prompts the user for input and handles status messages based on input type."""
    print("\n--- Project Chronos: The AI Archeologist ---")
    fragment = input("Enter the fragmented, Morse code, or non-English text: ")
    
    # Status messages help the user see what the program is recognizing
    is_morse = all(ch in ".- /" for ch in fragment.strip())
    is_english_slang = not is_morse and any(len(word) <= 4 for word in fragment.split())
    
    if is_morse:
        print("\n⏳ Detected potential Morse code. Sending to Gemini for decryption...")
    elif not is_english_slang and fragment != "":
        # Simple heuristic to guess if it might be non-English or just long plain text
        print("\n🌐 Sending text to Gemini for initial translation/reconstruction...")

    return fragment


# --- Core Gemini Functions ---

def initialize_gemini():
    """Initializes and returns the Gemini client using the environment variable."""
    try:
        # Check added for robust error handling
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable not found.")
            
        client = genai.Client()
        return client
    except Exception as e:
        print("Error initializing Gemini client. Make sure GEMINI_API_KEY is set and valid.")
        print(f"Details: {e}")
        sys.exit(1)


def reconstruct_text(client: genai.Client, fragment: str):
    """Handles decryption, translation, and slang reconstruction via a single Gemini call."""
    print("🤖 Analyzing, decrypting, translating, and reconstructing text...")
    prompt = RECONSTRUCTION_PROMPT.format(fragment)
    try:
        response = client.models.generate_content(
            model=RECONSTRUCTION_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except APIError as e:
        print(f"Gemini API Error during reconstruction: {e}")
        return None

def find_contextual_sources(client: genai.Client, reconstructed_text: str):
    """Uses Gemini with the Google Search tool to find and extract contextual sources."""
    print("🔍 Searching the web for contextual sources...")
    search_query = f"Provide definitions and context for all slang and cultural references in this sentence: '{reconstructed_text}'"
    try:
        response = client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=search_query,
            config=genai.types.GenerateContentConfig(
                tools=[{"google_search": {}}]
            )
        )
        sources = []
        if response.candidates and response.candidates[0]:
            candidate = response.candidates[0]
            if candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                if metadata.grounding_chunks:
                    print("✅ Found search results.")
                    for i, chunk in enumerate(metadata.grounding_chunks):
                        if i < 5 and chunk.web.uri:
                            sources.append(chunk.web.uri)
                        if len(sources) >= 5:
                            break
        if not sources:
            sources.append("No specific web sources found by the GEMINI AI model for the reconstructed text.")
        return sources
    except APIError as e:
        print(f"Gemini API Error during search grounding: {e}")
        return ["Error performing search."]

def generate_report(original_fragment: str, reconstructed_text: str, sources: list):
    """Formats and prints the final Reconstruction Report."""
    # Adjusted print width for better readability on standard terminals
    print("\n" + "="*70)
    print(" " * 19 + "--- PROJECT CHRONOS: RECONSTRUCTION REPORT ---")
    print("="*70 + "\n")
    print("[Original Fragment]")
    print(f"> {original_fragment}\n")
    print("[AI-Reconstructed Text]")
    print(f"> {reconstructed_text}\n")
    print("[Contextual Sources]")
    for source in sources:
        print(f"* {source}")
    print("\n" + "="*70)

def main():
    client = initialize_gemini()
    original_fragment = get_fragment_input()
    
    if not original_fragment:
        print("Input cannot be empty. Exiting.")
        return
        
    # All pre-processing (decryption/translation) and reconstruction happen here
    reconstructed_text = reconstruct_text(client, original_fragment)
    
    if not reconstructed_text:
        print("Could not complete reconstruction.")
        return
        
    contextual_sources = find_contextual_sources(client, reconstructed_text)
    
    generate_report(original_fragment, reconstructed_text, contextual_sources)

if __name__ == "__main__":
    main()
