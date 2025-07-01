import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googleapiclient.discovery import build
from streamlit_tags import st_tags
import google.generativeai as genai
import json
import textwrap

# 🔹 Configure Streamlit Page
st.set_page_config(layout="wide", page_title="ChefAI 🧑🍳", page_icon="🍳")

# 🔹 Set up Gemini API
GEMINI_API_KEY = "AIzaSyDbSrbKPbNK-sb3sx5tXMoPlN30Q3_P5o8" 
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 YouTube API Key
YOUTUBE_API_KEY = "AIzaSyC_mVSKJhciP_WXMe2PcUSEC4BoMHPfTXY" 

# 🔹 YouTube Search with Caching
@st.cache_data(ttl=3600)
def search_youtube(query, max_results=3):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            relevanceLanguage="en"
        )
        response = request.execute()
        return [{
            "title": item["snippet"]["title"],
            "url": f"https://youtu.be/{item['id']['videoId']}"
        } for item in response.get("items", [])]
    except Exception as e:
        st.error(f"Error searching YouTube: {str(e)}")
        return []

# 🔹 Recipe Formatting
def format_recipe(raw_text):
    try:
        sections = raw_text.split('\n')
        formatted = ""
        for section in sections:
            if section.startswith('title:'):
                formatted += f"## 🍴 {section[6:].strip()}\n\n"
            elif section.startswith('ingredients:'):
                items = [x.strip() for x in section[12:].split(';') if x.strip()]
                formatted += "### 📝 Ingredients\n" + "\n".join(f"- {item}" for item in items) + "\n\n"
            elif section.startswith('directions:'):
                steps = [x.strip() for x in section[11:].split(';') if x.strip()]
                formatted += "### 👩🍳 Instructions\n" + "\n".join(
                    f"{i+1}. {step}" for i, step in enumerate(steps)
                ) + "\n\n"
        return formatted
    except:
        return raw_text  

# 🔹 Load Models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-recipe-generation")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer), genai.GenerativeModel("gemini-1.5-flash")

generator, gemini = load_models()

# 🔹 Load Ingredients
with open("config.json") as f:
    cfg = json.load(f)

# 🔹 Main Interface
st.header("🍽️ ChefAI - Smart Recipe Generator & Cooking Assistant 🤖")

# 🔹 Recipe Generation Section
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🍲 Create Your Recipe")
        sampling_mode = st.selectbox("Generation Mode", ["Balanced", "Creative", "Precise"], 
                                    help="Control how creative the recipe generation should be")
        original_keywords = st.multiselect("Select Ingredients", cfg["first_100"], [],
                                         placeholder="Choose main ingredients...")
        custom_keywords = st_tags(label="Add More Ingredients", text='Press enter to add',
                                 suggestions=cfg["next_100"], maxtags=15)

    with col2:
        st.subheader("⚙️ Preferences")
        dietary_restrictions = st.multiselect("Dietary Needs", ["Vegetarian", "Vegan", "Gluten-Free"])
        max_length = st.slider("Recipe Complexity", 100, 500, 200)
        temperature = st.slider("Creativity Level", 0.1, 1.0, 0.7)

    all_ingredients = ", ".join(original_keywords + custom_keywords)
    
    if st.button("✨ Generate Recipe!", use_container_width=True):
        if not all_ingredients:
            st.error("Please select at least one ingredient!")
            st.stop()
            
        with st.spinner("Cooking up your recipe..."):
            try:
                generated = generator(
                    f"{dietary_restrictions} {all_ingredients}",
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1
                )
                raw_recipe = generated[0]['generated_text']
                formatted_recipe = format_recipe(raw_recipe)

                # Store context
                st.session_state["last_recipe"] = raw_recipe
                st.session_state["last_ingredients"] = all_ingredients
                
                # Display recipe
                st.subheader("📜 Your Custom Recipe")
                st.markdown(formatted_recipe)
                
                # YouTube Videos
                st.subheader("🎥 Cooking Videos")
                videos = search_youtube(f"{all_ingredients} recipe")
                if videos:
                    cols = st.columns(len(videos))
                    for idx, col in enumerate(cols):
                        with col:
                            st.video(videos[idx]["url"])
                            st.caption(videos[idx]["title"])
                else:
                    st.info("No videos found. Try different ingredients.")

            except Exception as e:
                st.error(f"Recipe generation failed: {str(e)}")

# 🔹 Food Chatbot
st.sidebar.header("💬 Cooking Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "content": "Hello! I'm ChefAI. Ask me about cooking, nutrition, or food science!"}]

# Display chat history
for message in st.session_state.messages[-5:]:
    st.sidebar.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

# Chatbot Input
chat_input = st.sidebar.text_input("Ask me anything...", key="chat_input")

if st.sidebar.button("Send", key="chat_send"):
    if chat_input:
        st.session_state.messages.append({"role": "user", "content": chat_input})
        
        # Include Recipe Context for Better Answers
        additional_context = ""
        if "last_recipe" in st.session_state and "last_ingredients" in st.session_state:
            additional_context = f"\nHere is the recipe and ingredients: {st.session_state['last_ingredients']}\n{st.session_state['last_recipe']}\n"

        # **Improved Prompt for Calorie & Serving Size Questions**
        if "calories" in chat_input.lower():
            prompt = f"""
            Estimate the total and per-serving calories for this recipe.
            Assume standard ingredient weights and provide a reasonable estimate.

            Recipe Ingredients:
            {st.session_state.get("last_ingredients", "Unknown")}
            
            Instructions:
            {st.session_state.get("last_recipe", "Unknown")}
            
            Keep the answer concise and useful.
            """
        elif "servings" in chat_input.lower():
            prompt = f"""
            Estimate how many servings this recipe makes based on standard portion sizes.

            Recipe Ingredients:
            {st.session_state.get("last_ingredients", "Unknown")}
            
            Instructions:
            {st.session_state.get("last_recipe", "Unknown")}
            
            Provide a reasonable estimate with common portion sizes.
            """
        else:
            prompt = f"{additional_context}\n{chat_input}"

        # Get Response from Gemini
        with st.spinner("Thinking..."):
            response = gemini.generate_content(prompt)
            bot_response = response.text if response else "Sorry, I couldn't find an answer."

        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.rerun()

st.sidebar.write("Ask about recipes, calories, storage tips, and more!")
