import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        self.name = 'cupid'
        self.user_ratings = {}
        self.prompted_disambiguate = False
        self.candidates = []
        self.recs_given = False
        self.current_sentiment = 0
        self.giving_recs = False
        self.recs = []
        self.current_rec_index = 0

        # shape: num_movies x num_users
        # value at i, j is the rating for movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')

        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        
        self.count_vectorizer = CountVectorizer(min_df=20, stop_words='english', max_features=1000) 

        self.train_logreg_sentiment_classifier()

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        cupid is a chatbot that recommends movies it thinks you'll <3 love <3
        
        you can tell cupid about movies you've enjoyed and movies you've hated, 
        and it will output new ones for you to watch based on these sentiments.
        
        cupid has access to multiple databases, including titles and ratings
        from MovieLens and reviews from Rotten Tomatoes.
        
        have fun chatting and discovering new movies with cupid! <3
        
        (to exit: write ":quit" or press ctrl-c)
        """

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = "hello! my name is cupid <3. i'll help you find new movies to watch and love if you tell me about movies you watched recently. But make sure to only tell me about one movie at a time and put its title in \"quotations\"!"
        
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "thank you for chatting with me! i hope you love the new movies! <3 <3 <3"

        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################
    # TODO:
    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input
        """
        if self.recs_given:
            self.user_ratings = {}
            self.prompted_disambiguate = False
            self.recs_given = False
            self.recs = []
            self.current_rec_index = 0
            return "feel free to ask for recommendations again by telling me about more movies you've seen! <3 (or enter :quit if you're done)"
    
        if len(self.user_ratings) < 5:
            if self.prompted_disambiguate:
                disam_candidates = self.disambiguate_candidates(line, self.candidates)
                
                if len(disam_candidates) > 1:
                    self.candidates = disam_candidates
                    candidate_string = '\n'.join([self.titles[c][0] for c in self.candidates])
                    return f"sorry, i still don't know which movie you're referring to:\n{candidate_string}"
                elif len(disam_candidates) == 0:
                    candidate_string = '\n'.join([self.titles[c][0] for c in self.candidates])
                    return f"sorry, i still don't know which movie you're referring to:\n{candidate_string}"
                
                self.candidates = disam_candidates
                self.prompted_disambiguate = False
            else:
                movie = self.extract_titles(line)
                
                if not movie:
                    emotion_response = self.handle_emotions(line)
                    
                    if emotion_response:
                        if emotion_response[0]:
                            return f"glad to hear you're {emotion_response[1]} <3 hopefully my movie recommendations keep you feeling that way!"
                        else:
                            return f"sorry to hear you're {emotion_response[1]} :( i hope my movie recommendations can cheer you up <3"
                        
                    return "i don't understand! please try again <3"
                
                movie = movie[0]
                self.current_sentiment = self.predict_sentiment_statistical(line)        
                self.candidates = self.find_movies_idx_by_title(movie)
                
                if len(self.candidates) > 1:
                    candidate_string = '\n'.join([self.titles[c][0] for c in self.candidates])
                    self.prompted_disambiguate = True
                    return f"which movie are you referring to:\n{candidate_string}"
                elif len(self.candidates) == 0:
                    return "sorry, i couldn't find that movie in my database :( please try again!"
            
            self.user_ratings[self.candidates[0]] = self.current_sentiment

            if len(self.user_ratings) < 5:
                movie_name = self.titles[self.candidates[0]][0]
                if self.current_sentiment == 1:
                    return f"awesome, {movie_name} is great! tell me about another movie <3"
                elif self.current_sentiment == 0:
                    return f"so that's your opinion on {movie_name}... tell me about another movie <3"
                else:
                    return f"you're not a fan of {movie_name}, huh? tell me about another movie <3"
        
        self.recs = self.recommend_movies(self.user_ratings, 10) 
           
        if self.giving_recs:
            line = re.sub(r'\W', '', line.lower())  
            self.current_rec_index += 1
            if self.current_rec_index > 9:
                self.giving_recs = False
                self.recs_given = True
                return self.process("")
            yes = ["yes", "yea", "yeah", "yah", "yuh", "y"]
            no = ["no", "nah", "nope", "naur", "nay", "n"]
            if line in no:
                self.giving_recs = False
                self.recs_given = True
                return self.process("")
            elif line in yes:
                return f"lovely! i recommend {self.recs[self.current_rec_index]} <3 want another recommendation?"

        self.giving_recs = True
        return f"lovely! i recommend {self.recs[self.current_rec_index]} <3 want another recommendation?"
            
    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        pattern = re.compile("\"[^\"]+\"")
        titles = re.findall(pattern, user_input)
        
        # Remove leading and trailing whitespace
        # Only include non-empty strings between quotes
        titles = [f"{title[1:-1].strip()}" for title in titles if title[1:-1].strip() != ""]
        
        return titles
    
    def find_movies_idx_by_title(self, title:str) -> list:
        """Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think 
              of a more concise approach 
        """
        indices = []
        
        no_punc = re.compile("\w+")
        tokens = re.findall(no_punc, title)
        
        for i, entry in enumerate(self.titles):
            for tok in tokens:
                if not re.search(r'\b{}\b'.format(tok.lower()), entry[0].lower()):
                    break
            else:
                indices.append(i)
                                                       
        return indices

    def disambiguate_candidates(self, clarification:str, candidates:list) -> list: 
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        indices = []

        no_punc = re.compile("\w+")
        tokens = re.findall(no_punc, clarification)

        for c in candidates:
            title_tokens = [t.lower() for t in re.findall(no_punc, self.titles[c][0])]
            for tok in tokens:
                if tok.lower() not in title_tokens:  
                    break
            else:
                indices.append(c)

        return indices

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment. 
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count) 
        and negative sentiment category (neg_tok_count)

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """                                              
        tokens = re.findall(r"\w+" , user_input)
        counts = Counter()

        for token in tokens:
            token = token.lower()
            if token in self.sentiment:
                if self.sentiment[token] == 'pos':
                    counts['pos'] += 1
                else:
                    counts['neg'] += 1

        if counts['neg'] > counts['pos']:
            return -1
        elif counts['neg'] < counts['pos']:
            return 1
        
        return 0 
    
    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        texts, y = util.load_rotten_tomatoes_dataset()
        texts = [text.lower() for text in texts]
        y[y == "Rotten"], y[y == "Fresh"] = -1, 1
        y = y.astype('int')

        texts_array = self.count_vectorizer.fit_transform(texts).toarray()
        
        self.model = sklearn.linear_model.LogisticRegression(penalty=None)
        self.model.fit(texts_array, y)

    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        user_input = user_input.lower()
        
        input_array = self.count_vectorizer.transform([user_input]).toarray()
        
        if sum([abs(val) for val in input_array[0]]) == 0:
            return 0
        
        return self.model.predict(input_array)[0]


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        user_ratings_all = [user_ratings[i] if i in user_ratings else 0 for i in range(len(self.ratings))]
        
        # returns list of movie indices corresponding to movies in ratings_matrix
        recs = util.recommend(user_ratings_all, self.ratings, num_return)
        
        return [self.titles[rec][0] for rec in recs]


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def handle_emotions(self, user_input:str): 
        """Identify and respond to user input expressing a particular
        emotion.
        
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (int, str) : a sentiment value for the specified emotion,
                           and the word for the emotion expressed
                - 1 is a positive sentiment
                - 0 is a negative sentiment
        
        Note: This function uses the first emotion it finds in the user input.
        """
        positive = [word.strip() for word in open('data/emotions_positive.txt', 'r').read().split('\n')]
        negative = [word.strip() for word in open('data/emotions_negative.txt', 'r').read().split('\n')]
        
        for word in user_input.split():
            word = re.sub(r'\W', '', word.lower())
            if word in positive:
                return (1, word)
            elif word in negative:
                return (0, word)
        
        return None

    def handle_movie_titles_with_articles(self, user_input: str) -> list:
        """The main functionality for this function is already in 
        'find_movies_idx_by_title' starting on line 199.
        
        This function is not explicitly included in process() because
        the code below is for testing purposes, i.e. demonstrating
        that 'find_movies_idx_by_title' indeed does return the correct 
        indices of movies that start with an article. However, 
        'find_movies_idx_by_title' is used in process().
        """
        titles = self.extract_titles(user_input)
        movie_indices = [idx for title in titles for idx in self.find_movies_idx_by_title(title)]
        
        return [self.titles[i][0] for i in movie_indices]
    
    def extract_titles_no_quote_wrong_cap(self, user_input: str) -> list:
        """Extract a movie title from user input even if it is not contained
        within quotation marks and has incorrect capitalization.
        
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles_no_quote_wrong_cap("qwertyuiop")
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles_no_quote_wrong_cap(
                                            'I liked "The Notebook" a lot.')
          print(potential_titles) // prints ["Notebook, The"]

        Example 3: 
          potential_titles = chatbot.extract_titles_no_quote_wrong_cap(
                                            "I liked 10 things i HATE AbouT yOU")
          print(potential_titles) // prints ["10 Things I Hate About You"] 
          
        Note: This function will always match the movie explicitly specified 
        by the user input, and thus does extract the correct movie. However,
        the implementation is sometimes more general, i.e. it will identify
        movies that are not explicitly specified. This can later be handled
        in disambiguate_candidates.
        
        Example:
          potential_titles = chatbot.extract_titles_no_quote_wrong_cap(
                                            "I liked The Hunger Games")    
          print(potential_titles) // prints ["The Hunger Games", "Hunger", "The Hunger"]                 
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text
        """
        titles = []
        
        # remove punctuation and articles from user input
        user_input = " ".join(re.findall(r"\w+", user_input.lower())).lower()
        user_input = re.sub(r"\b(a|an|the)\b", '', user_input).strip()
        
        for movie in self.titles:
            original_title = movie[0][:-7] # remove release date from title
            
            # remove punctuation and articles from movie title
            movie = " ".join(re.findall(r"\w+", original_title.lower()))        
            movie = re.sub(r"\b(a|an|the)\b", '', movie).strip()
            
            if re.search(r'\b{}\b'.format(re.escape(movie)), user_input):
                titles.append(original_title)
        
        return list(set(titles))


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



