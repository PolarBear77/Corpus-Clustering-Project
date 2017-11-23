# coding: utf-8
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def draw_cloud(text_string, x=20, y=20):
    wordcloud = WordCloud(max_font_size=400).generate(text = text_string) # lower max_font_size
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.figure(figsize=(x,y))
    plt.axis("off")
    plt.show()
    
# For showing in a new window  
#     wordcloud = WordCloud().generate(text = text_string)
#     image = wordcloud.to_image()
#     image.show()

