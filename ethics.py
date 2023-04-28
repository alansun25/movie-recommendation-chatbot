"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

The potential for users of our chatbot to anthropomorphize it is relatively low. Our chatbot responses 
were made to be movie-specific, and its responses to a user’s emotional input are hard-coded for positive
and negative adjectives. With each interaction, our users are constantly reminded that our chatbot is a 
non-human specifically designed for recommending movies. Its responses are not personalized, adaptable, or 
emotional enough for anthropomorphizing. 

Successfully anthropomorphized chatbots induce a false sense of natural and personal connection, which 
encourages perceived trustworthiness. By creating this relationship, these chatbots would be able to 
outperform and replace human agents in performing certain tasks. In an industrial context, this could 
provide companies with cost savings and service efficiencies with 24/7 chatbot assistance. On a personal 
level, complex anthropomorphized chatbots may display “sentience” that may arouse alarm. Although truly 
“sentient” AI is still just sci-fi, fake sentience can still encourage the tendency to become attached
to a machine as if it were a real person. This risks psychological entanglement with technology that 
may be detrimental for mental health. Thus, it is important that precautions are taken to ensure that 
technology doesn’t become psychologically harmful. In order to distinguish chatbot and human responses, 
designers can make chatbot replies extremely generic and rule-based, clearly displaying inhuman traits. 
Chatbots can also be trained to have domain knowledge in only a certain field, and be prevented from 
retaining information in conversation that is irrelevant to that domain. For example, if a chatbot’s 
purpose is to recommend movies, a user shouldn’t be able to impose personal political beliefs on it, 
they should only be able to tell it about movies.

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Because our chatbot is so task-specific to recommending movies, there isn’t a real danger in 
releasing personal or private information. Our chatbot resets the gathered user rating of movies 
and the outputted recommendations after each round of learning user sentiment from 5 movies. On 
the surface, the stored information is wiped clean. However, in general, designers of chatbots 
can implement stronger security measures, such as encryption, especially if the chatbot is made 
to assist with personal tasks. The model can be also trained to refuse and censor oddly specific 
personal or dangerous requests. On the user side, they should be more careful not to enter any 
confidential information wherever it’s not necessary or not on a reliable and tested platform. 

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

To mitigate the issue of widespread unemployment, we need to be aware of the benefits and disadvantages 
that automation poses. Usually, these advanced systems excel at a specific set of tasks, but their 
substitution of an entire human occupation may fail due to the lack of emotional and reasonable adaptability. 
With the upbringing of new technologies, there will be an emergence of different types of occupations that
require human intervention, such as filtering bias from training data and ensuring long-term stability and 
versatility of the technology. Employees at technology companies should upkeep this hiring of human employees,
to coexist with automation instead of allowing it to take over entirely. Government officials should make 
policies that protect the unequal access to new innovations that arise when developed countries advance 
faster than others. Special laws should also be made for automation mistakes (i.e. in medical fields) - where 
does the responsibility lie? 

"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """

Rule-based systems are useful when there are lower volumes of data and the rules are relatively simple to 
write. They are reliable and return results with high precision. However, they are also inflexible, since it 
is tedious to list out all the rules. Machine learning approaches are more accurate, adaptable, and quick, 
but the system requires a lot of training data. Deep learning approaches have better precision, recall, 
adaptability, and applicability, since there are many nested layers of neural networks. However, there is a 
lack of predictability, interpretability, and control because there is less human dependency. The training 
is also very expensive, relying on large amounts of hardware.

"""