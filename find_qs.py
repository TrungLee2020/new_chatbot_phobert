import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplexQuestionHandler:
    def __init__(self, data_path):
        # Load NLTK resources
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        # Load CSV data
        self.data = pd.read_csv(data_path)
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['question'])
        # Initialize Sentence Transformer model for embeddings
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Split patterns for complex questions
        self.split_patterns = [
            r'\s+và\s+',  # "và"
            r'\s+hoặc\s+',  # "hoặc"
            r'\s+hay\s+',  # "hay"
            r'\s+cùng\s+với\s+',  # "cùng với"
            r'\s+đồng thời\s+',  # "đồng thời"
            r'\?',  # Question mark
            r'\s+như thế nào\s+',  # "như thế nào"
            r'\s+bằng cách nào\s+',  # "bằng cách nào"
            r'\s+làm sao\s+',  # "làm sao"
            r'\s+là gì\s+'  # "là gì"
        ]

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def expand_question_with_synonyms(self, question):
        words = word_tokenize(question)
        expanded_words = []
        for word in words:
            synonyms = self.get_synonyms(word)
            expanded_words.extend(synonyms)
            expanded_words.append(word)  # include the original word as well
        expanded_question = ' '.join(expanded_words)
        return expanded_question

    def find_standard_question(self, user_question):
        logger.info(f"Find the standard question from: {user_question}")
        expanded_question = self.expand_question_with_synonyms(user_question)
        # Calculate TF-IDF for expanded user question
        user_tfidf = self.tfidf_vectorizer.transform([expanded_question])
        # Compute cosine similarity between expanded user question and standard questions
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        # Get the index of the most similar question
        best_match_idx = cosine_similarities.argmax()
        # Verify with vector embeddings for better accuracy
        user_embedding = self.sentence_model.encode(user_question, convert_to_tensor=True)
        standard_embeddings = self.sentence_model.encode(self.data['question'].tolist(), convert_to_tensor=True)
        embedding_cos_scores = util.pytorch_cos_sim(user_embedding, standard_embeddings)
        # Combine TF-IDF and embedding scores
        combined_scores = cosine_similarities + embedding_cos_scores.cpu().numpy().flatten()
        best_combined_match_idx = combined_scores.argmax()
        return self.data.iloc[best_combined_match_idx]

    def analyze_question(self, user_question, context):
        standard_question = self.find_standard_question(user_question)
        new_context_id = standard_question['context_id']
        if new_context_id != context['context_id']:
            context['context_id'] = new_context_id
            # context['doc_ids'] = standard_question['doc_ids']
            context['question_id'] = standard_question['question_id']

        # std_answers = []
        # doc_ids = []
        # doc_names = []
        # final_response = []
        # for idx, _ in standard_question:
        #     similar_question = self.data.iloc[idx]
        #     std_answers.append(similar_question['answer'])
        #     doc_ids.append(similar_question['doc_id'])

        return {
            "user_question": user_question,
            "std_question": standard_question['question'],
            "question_id": standard_question['question_id'],
            "context_id": context['context_id'],
            "doc_ids": context['doc_ids']
        }

    def split_complex_question(self, question):
        pattern = '|'.join(self.split_patterns)
        parts = re.split(pattern, question)

        # Clean and filter the parts
        cleaned_parts = [part.strip() for part in parts if part.strip()]

        return cleaned_parts

    def process_complex_question(self, user_question):
        # Split the complex question into smaller parts
        question_parts = self.split_complex_question(user_question)

        context = {"context_id": None, "doc_ids": []}
        results = []
        question_ids = []
        std_questions = []

        for part in question_parts:
            result = self.analyze_question(part, context)
            results.append(result)
            question_ids.append(result['question_id'])
            std_questions.append(result['std_question'])

        logger.info(f"Question ids: {question_ids}")
        logger.info(f"Standard question: {std_questions}")

        return {
            "user_question": user_question,
            "question_ids": question_ids,
            "std_questions": std_questions,
            "context_id": context['context_id'],
            "doc_ids": context['doc_ids']
        }

# if __name__ == "__main__":
#     handler = ComplexQuestionHandler('data_sources/db.csv')
#     user_question = "tra cứu đơn hàng và hàng cấm"
#     result = handler.process_complex_question(user_question)
#     print(f"User Question: {result['user_question']}")
#     print(f"Question IDs: {result['question_ids']}")
#     # print(f"Context ID: {result['context_id']}")
#     # print(f"Doc IDs: {result['doc_ids']}")
#     print(f"Standard Questions: {result['std_questions']}")
