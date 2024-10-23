import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("Sarwar242/autotrain-fake-reviews-labelling-37433101195")
tokenizer = AutoTokenizer.from_pretrained("Sarwar242/autotrain-fake-reviews-labelling-37433101195")

# Mapping between abbreviated and full forms
class_mapping = {"CG": "Fake Review", "OR": "Original Review"}

# Streamlit app
def main():
    st.title('Fake Reviews Classification')

    # User input text box
    user_input = st.text_area('Enter the review text:', '')

    if st.button('Classify') and user_input:
        # Tokenize the user input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Make a prediction using the pre-trained model
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        predicted_class = logits.argmax().item()

        # Get the class label and confidence
        class_label = model.config.id2label[predicted_class]
        confidence = logits.softmax(dim=1).max().item()

        # Display the result
        full_class_label = class_mapping.get(class_label, class_label)
        st.write('Predicted Class:', full_class_label)
        st.write('Confidence:', confidence)

# Run the app
if __name__ == '__main__':
    main()
