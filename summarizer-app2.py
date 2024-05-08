import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Define a function to generate summaries
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=False, no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    st.title("Article Summarizer")

    # Text input for the user to input the article
    article = st.text_area("Enter your article:")

    # Button to generate the summary
    if st.button("Generate Summary"):
        if article:
            summary = generate_summary(article)
            # Display the original text and the summary
            st.subheader("Original Text:")
            st.write(article)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
