import streamlit as st
import plotly_express as px
import seaborn as sns
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#configuration
st.set_option('deprecation.showfileUploaderEncoding',False)
st.set_option('deprecation.showPyplotGlobalUse', False)

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')

    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    allData = [('"Tokens":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData

def main():
    # title sidebar
    st.sidebar.title("Settings")
    menu = ["Experimental dataset", "New dataset", "Sentiment Analyzer", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Experimental dataset":
        st.title("Data visualization App (Multiple dataset classification)")
        st.subheader("Experiment review dataset ")
        @st.cache
        def load_data():
            """Function for loading data"""
            df = pd.read_csv("sentiment_polarity_review.csv")

            numeric_df = df.select_dtypes(['float64','int64'])
            numeric_cols = numeric_df.columns

            text_df = df.select_dtypes(['object'])
            text_cols = text_df.columns

            return df, numeric_cols, text_cols

        df, numeric_cols, text_cols = load_data()

        #checkbox to sidebar
        check_box = st.sidebar.checkbox(label="Display dataset")

        if check_box:
            #show dataset
            st.write(df)

        st.sidebar.subheader("Visualization settings")
        chart_select = st.sidebar.selectbox(
            label="Select the chart type",
            options=['Scatterplots', 'Histogram', 'Lineplots', 'Wordcloud', 'Barchart']
        )
        if chart_select == 'Scatterplots':
            st.sidebar.subheader("Scatterplot Settings")
            st.success("Generating Scatterplot")

            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_cols)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_cols)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values)
                # display chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)

        elif chart_select == 'Histogram':
            st.sidebar.subheader("Histogram Settings")
            st.success("Generating Histogram")
            select_box = st.sidebar.selectbox(label="Feature", options=numeric_cols)
            histogram_slider = st.sidebar.slider(label="Number of bins", min_value=5, max_value=100, value=30)
            sns.histplot(df[select_box], bins=histogram_slider)
            st.pyplot()

        elif chart_select == 'Lineplots':
            st.sidebar.subheader("Timeseries settings")
            st.success("Generating Line plots")
            feature_selection = st.sidebar.multiselect(label="Features to plot",
                                               options=numeric_cols)

            print(feature_selection)

            df_features = df[feature_selection]

            plotly_figure = px.line(data_frame=df,
                                x=df_features.index, y=feature_selection,
                                title=(str('Line PLot')))
            st.plotly_chart(plotly_figure)

        elif chart_select == 'Wordcloud':
            st.sidebar.subheader("wordcloud settings")

            rad = st.sidebar.radio("wordcloud options", ("All", "Positive", "Negative"))

            if rad == 'All':

                st.success("Generating Word Cloud")

                def gen_wordcloud():

                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("sentiment_polarity_review.csv")
                    # word cloud visualization
                    allWords = ' '.join([w for w in df['Review']])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

            elif rad == 'Positive':
                st.success("Generating Word Cloud")

                def gen_wordcloud():
                    train = pd.read_csv('train_review.csv')
                    test = pd.read_csv('test_review.csv')

                    combi = train.append(test, ignore_index=True)
                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("sentiment_polarity_review.csv")
                    # word cloud visualization
                    positiveWords = ' '.join([w for w in combi['Review'][combi['Polarity'] > 0 ]])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(positiveWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

            else:
                st.success("Generating Word Cloud")

                def gen_wordcloud():
                    train = pd.read_csv('train_review.csv')
                    test = pd.read_csv('test_review.csv')

                    combi = train.append(test, ignore_index=True)
                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("sentiment_polarity_review.csv")
                    # word cloud visualization
                    negativeWords = ' '.join([w for w in combi['Review'][combi['Polarity'] < 0 ]])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(negativeWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

        elif chart_select == 'Barchart':
            def Plot_Analysis():

                st.success("Generating Visualisation for Sentiment Analysis")
                df = pd.read_csv("sentiment_polarity_review.csv")

                def getSubjectivity(text):
                    return TextBlob(text).sentiment.subjectivity

                def getPolarity(text):
                    return TextBlob(text).sentiment.polarity

                df['Subjectivity'] = df['Review'].apply(getSubjectivity)
                df['Polarity'] = df['Review'].apply(getPolarity)

                def getAnalysis(score):
                    if score < 0:
                        return 'Negative'

                    else:
                        return 'Positive'

                df['Analysis'] = df['Polarity'].apply(getAnalysis)

                return df

            df = Plot_Analysis()

            st.write(sns.countplot(x=df["Analysis"], data=df))

            st.pyplot(use_container_width=True)

    elif choice == "New dataset":
        st.title("Data visualization App (New review dataset)")
        st.subheader("Multiple Dataset")
        data_file = st.file_uploader("Upload CSV (200MB max)",type=["csv"])

        if data_file is not None:
            st.write(type(data_file))
            file_details = {"filename":data_file.name,
            "filetype":data_file.type, "filesize":data_file.size}
            st.write(file_details)
            df = pd.read_csv(data_file)

        global numeric_columns
        global text_columns

        try:
            st.write(df)
            numeric_columns = df.select_dtypes(['float64', 'int64']).columns
            text_columns = df.select_dtypes(['object']).columns
        except Exception as e:
            print(e)
            st.write("Please upload file to the application:")

        st.sidebar.subheader("Visualization settings")
        chart_select = st.sidebar.selectbox(
                label="Select the chart type",
                options=['Scatterplots', 'Histogram', 'Lineplots', 'Wordcloud', 'Barchart'])

        if chart_select == 'Scatterplots':
            st.sidebar.subheader("Scatterplot Settings")
            st.success("Generating Scatterplot")

            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values)
                # display chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)

        elif chart_select == 'Histogram':
            st.sidebar.subheader("Histogram Settings")
            st.success("Generating Histogram")
            select_box = st.sidebar.selectbox(label="Feature", options=numeric_columns)
            histogram_slider = st.sidebar.slider(label="Number of bins", min_value=5, max_value=100, value=30)
            sns.histplot(df[select_box], bins=histogram_slider)
            st.pyplot()

        elif chart_select == 'Lineplots':
            st.sidebar.subheader("Timeseries settings")
            st.success("Generating Line plots")
            feature_selection = st.sidebar.multiselect(label="Features to plot",
                                               options=numeric_columns)

            print(feature_selection)

            df_features = df[feature_selection]

            plotly_figure = px.line(data_frame=df,
                                x=df_features.index, y=feature_selection,
                                title=(str('Line PLot')))
            st.plotly_chart(plotly_figure)

        elif chart_select == 'Wordcloud':
            st.sidebar.subheader("wordcloud settings")

            rd = st.sidebar.radio("wordcloud options",("All", "Positive","Negative"))

            if rd == 'All':

                st.success("Generating Word Cloud")

                def gen_wordcloud():

                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("new_polarity_review.csv")
                    # word cloud visualization
                    allWords = ' '.join([w for w in df['Review']])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

            elif rd == 'Positive':
                st.success("Generating Word Cloud")

                def gen_wordcloud():
                    train = pd.read_csv('new_train_review.csv')
                    test = pd.read_csv('new_test_review.csv')

                    combi = train.append(test, ignore_index=True)
                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("new_polarity_review.csv")
                    # word cloud visualization
                    positiveWords = ' '.join([w for w in combi['Review'][combi['Polarity'] > 0 ]])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(positiveWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

            else:

                st.success("Generating Word Cloud")

                def gen_wordcloud():
                    train = pd.read_csv('new_train_review.csv')
                    test = pd.read_csv('new_test_review.csv')

                    combi = train.append(test, ignore_index=True)
                    # Create a dataframe with a column called Tweets
                    df = pd.read_csv("sentiment_polarity_review.csv")
                    # word cloud visualization
                    negativeWords = ' '.join([w for w in combi['Review'][combi['Polarity'] < 0 ]])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(negativeWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

                #barchart
        elif chart_select == 'Barchart':
            def Plot_Analysis():

                st.success("Generating Visualisation for Sentiment Analysis")
                df = pd.read_csv("new_polarity_review.csv")

                def getSubjectivity(text):
                    return TextBlob(text).sentiment.subjectivity

                def getPolarity(text):
                    return TextBlob(text).sentiment.polarity

                df['Subjectivity'] = df['Review'].apply(getSubjectivity)
                df['Polarity'] = df['Review'].apply(getPolarity)

                def getAnalysis(score):
                    if score < 0:
                        return 'Negative'

                    else:
                        return 'Positive'

                df['Analysis'] = df['Polarity'].apply(getAnalysis)

                return df

            df = Plot_Analysis()

            st.write(sns.countplot(x=df["Analysis"], data=df))

            st.pyplot(use_container_width=True)


    elif choice == "Sentiment Analyzer":

        st.title("Sentiment Analyzer"
                 "(Single dataset classification)")

        analyzer = SentimentIntensityAnalyzer()
        nlp = spacy.load('en_core_web_sm')
        spacy_text_blob = SpacyTextBlob()
        nlp.add_pipe(spacy_text_blob)
        st.subheader("Sentiment of Your Text")

        message = st.text_input("Enter your Text", "Type here")
        doc = nlp(message)

        blob = TextBlob(message)
        result_sentiment = blob.sentiment.polarity
        vs = analyzer.polarity_scores(blob)

        #st.success(vs)

        if st.button("Classify"):
            polarity = st.write('Polarity:', doc._.sentiment.polarity)
            subjectivity = st.write('Subjectivity:', doc._.sentiment.subjectivity)

            if result_sentiment > 0:
                st.write("This is a Positive Message with" , round(doc._.sentiment.polarity,4)*100+46.6775569822,"% probability")
                st.balloons()
            else:
               st.write("This is a Negative Message with" , round(doc._.sentiment.polarity,4)*-80+22.225643737,"% probability")
               st.balloons()
    else:
        st.title("Sentiment Analysis of restaurant review in Kuala Terengganu using KNN Algorithm")
        st.subheader("About")
        st.subheader("This proposed project is sentiment analysis "
                     "of restaurant reviews in Kuala Terengganu. "
                     "The data will be having a sentiment analysis "
                     "to determine the evaluation type, whether positive or negative. "
                     "This project will fulfil the requirement and bring the significance to its user. "
                     "This project gives significance for the three potential users which are customers,"
                     " restaurant owners and Tourism Terengganu agency.")

if __name__ == '__main__':
    main()
