import numpy as np
import pandas as pd

# LDA library
from sklearn.decomposition import LatentDirichletAllocation

# GridSearchCV library
from sklearn.model_selection import GridSearchCV

# build LDA model
def buildLDA(nlp, docs, relevant):
    lda_grid = LatentDirichletAllocation(max_iter=15, learning_method='batch', learning_offset=50., random_state=0)

    # init Grid Search Class
    search_params = {'n_components': [5], 'learning_decay': [.3, .5]}
    model = GridSearchCV(lda_grid, param_grid=search_params)
    model.fit(nlp)
    
    # Best LDA model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity (lower is better)
    print("Model Perplexity: ", best_lda_model.perplexity(nlp))
    # Coherence
    
    # Create Document — Topic Matrix
    lda_output = best_lda_model.transform(nlp)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(docs))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    
    df_document_topic['isRelevant'] = relevant

    # Apply Style
    df_document_topics = df_document_topic.style.applymap(color_green).applymap(make_bold)
    
    return lda_output, df_document_topics, df_document_topic, model.best_score_, best_lda_model.perplexity(nlp)


# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

