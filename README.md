## Lindy List
##### Swing Music Dance Classifier

LindyList is a music classifier which determines the predominant dance style(s) of a given jazz-era song. Dance styles in scope:

* Lindy Hop
* Slow Swing/Drag Blues
* Balboa
* Charleston
* St Louis and Collegiate Shag

![Lindy Hop](img/jean_lindy.gif?raw=true "Lindy Hop 1")

#### Introduction
Auto-tagging music genre has been explored numerous times in recent years using deep learning. Many of the attempts to classify music focus on a wide variety of music genres. This project embraces some of those music genre classification techniques and applies them to swing music  to tag dance style(s) which can later be used for song discovery and playlist recommendations.

The LindyList is for jazz-era music buffs, jazz musicians, DJs, performers, social swing dancers and film/TV sound design. Dance-style knowledge would typically be acquired from years of exposure to a variety of jazz music and dances styles. With this project, I intend to make swing music discovery more accessible and engaging from the dancer’s perspective.

#### The Dataset
3000 unique songs, were randomly selected and downloaded from http://www.jazz-on-line.com/. The music spans a period between 1925-1959. Since dance style labels did not exist for this dataset, I manually labelled dance style tags (yes you read that right, it took me about 50 hours if you're curious). The data includes annotations such as song likeability and recording quality. For this reason the classifier is biased towards my own preferences and ideas about sub-genres in swing music.

This totaled 1376 swing-danceable songs to work with for training and validation testing.

#### Audio Feature Extraction
From the mp3s I extracted: harmonic & percussive tempo, beats per song, 12 pitch prominence scores, relative root mean square energy from each song for the basic ensemble model.

For each song, I also generated multiple 30 second samples of the log-power mel-spectrograms for training the convolutional neural network models. The idea was to come up with a basic baseline model to for the neural nets to improve upon.

![Lindy Hop](img/mel-specs.png?raw=true "Lindy Hop 1")

#### Classifier Modeling
Machine learning models were prototyped and compared for best accuracy scores.

The following methods were explored for the baseline ensemble model:
* Random Forest Classifier
* Gradient Boosted Classifier
* K Nearest Neighbors

The following neural net models were explored:
* Convolutional Neural Nets
* Convolutional Recurrent Neural Nets

Certain classes were difficult to predict and the slow swing/ drag blues category was eventually left out of the neural net models due to too much variation in the sub-category.

#### Results
The ensemble model, which consisted of two differently optimized random forest classifiers and one k nearest neighbors classifier achieved 50% accuracy across 5 classes and 60.9% accuracy across 4 classes (when excluding slow swing /drag blues)

![Ensemble Results](img/Ensemble_results.png?raw=true "Results Model 1")

... while 81% was  achieved using a 4-layer deep convolutional recurrent neural network across 4 classes.

![CRNN Results](img/CRNN_results.png?raw=true "Results Model 2")

** It worked! Great validation results! Let's build the app! But not so fast... **

#### Problems in Production
When testing my model with songs sampled from online sources (e.g youtube.com), the effects of compression on the song degraded the contrast in pitch amplitude of the mel-spectrograms compared to the jazz-on-line.com mp3s.
This resulted in a skewed classification bias towards the mid-tempo dance style, Lindy Hop.

#### Plans for the Future
I have plans to incorporate Slow Swing/Drag Blues category into the neural net training and optimize learning rates, activation function parameters appropriately. I also believe auditing the quality of manual labelling for this category may also help since the more specific the better neural nets perform. I intend to experiment with different pre-processing of the mel-spectrograms to counteract the compression effects depending on the source type.

#### Stay tuned for the app!
My end goal is to have a working web app and mobile app for users to identify song dance style and suggested similar style playlist of songs from the database. Watch out for this space!
