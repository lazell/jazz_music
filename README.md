## Lindy List
##### Swing Dance Music Classifier

LindyList is a music classifier that analyzes a jazz-era
song to determine the most suitable style(s) of swing dance
from the five predominant styles:

* Lindy Hop
* Slow Swing/Drag Blues
* Balboa
* Charleston
* St Louis and Collegiate Shag

![Lindy Hop](img/jean_lindy.gif?raw=true "Lindy Hop 1")

#### Introduction
Auto-tagging music genre has been an active area of deep learning research in recent years. The LindyList, uses many of those same approaches, but instead of classifies by the most suitable style of swing dance from the five most common styles danced today. The prediction(s) can be used as a feature to improve song discovery and quality of of playlist and song recommendations.

The LindyList is for jazz-era music buffs, jazz musicians, DJs, performers, social swing dancers and film/TV sound design. Dance-style knowledge would typically be acquired from years of exposure to a variety of jazz music and dances styles. With this project, I intend to make swing music discovery more accessible and engaging from the dancer’s perspective.

#### The Dataset
The data set consists of 3000 unique songs, recorded between 1926-1959. Each was randomly selected
and downloaded from [Jazz On Line](https://http://www.jazz-on-line.com/), a public domain jazz
music website. Since labels for dance styles did not exist for
this dataset, I manually labeling songs by dance style and annotating each song with notes on whether a song was swing danceable, whether it
was particularly infectious, and whether the quality of the
recording was good. Since I labelled the dataset myself,
the dataset is undoubtedly biased toward my own
in swing dancing interpretations. Out of the 3000
songs, 1376 were “swing-danceable,” and were used for
training and validation testing.

#### The Approach
This project has two components:  the Ensemble
model and the Neural Net model. To build a baseline to
model, I first took the Ensemble
model approach. I extracting from each song harmonic and;
percussive tempo, total beat count per song, 12 pitch prominence
scores, and relative root mean square energy values. I then
prototyped various machine learning models and compared
for accuracy:

• Random Forest Classifier
• Gradient Boosted Classifier
• K Nearest Neighbors

From there, I generated multiple 30-second samples of each song and generated the log-power mel-spectrogram arrays to use as training inputs for the artificial neural network models:

• Convolutional Neural Nets
• Convolutional Recurrent Neural Nets

![Lindy Hop](img/mel-specs.png?raw=true "Lindy Hop 1")

In doing so, I discovered that certain classes were difficult to
predict and eventually decided to remove the slow
swing/drag blues category from the neural network models since it had
too much variability in the sub-category.


## Results

#### Ensemble Model Results
The ensemble model, which consisted of two differently optimized random forest classifiers and one k nearest neighbors classifier achieved 50% accuracy across 5 classes and 60.9% accuracy across 4 classes (when excluding slow swing /drag blues)

![Ensemble Results](img/Ensemble_results.png?raw=true "Results Model 1")

#### Neural Net Model Results
Using a Convolutional Recurrent Neural Network I was able to achieve 81% using a 4-layer deep convolutional recurrent neural network across the same 4 classes.

More importantly, the recall results for individual dance styles had increased to over 52%, with Charleston and Shag styles performing particularly well compared to the ensemble baseline.

By using a neural network model, I was able to reduce prediction time to approximately the tenth of the time it takes the ensemble model predict (both times are inclusive of audio pre-processing).

![CRNN Results](img/CRNN_results.png?raw=true "Results Model 2")


### Real World Testing
When testing LindyList performance with songs sampled from various online
sources, I discovered that the effects of compression in
newer music degraded the contrast in pitch amplitude of the
mel-spectograms to sufficient degree to skew LindyList
recommendations toward Lindy Hop, the more mid-tempo
dance style.

### Plans for the Future
I have plans to incorporate Slow Swing/Drag Blues category into the neural net by splitting the labels into different sub-genres of blues dancing, since the data exists in the dataset for this level of subcategorization.

To make LindyList more accessible, I plan to develop a mobile app to enable users to identify swing dance styles and suggest playlists of similar style songs from Jazz On Line.

In order for the app to be production ready, the plan is to experiment with different pre-processing techniques of the mel-spectrograms to counteract any compression effects and reduce noise in audio samples so that the model generalizes well to any source type (e.g. compressed online mp3s, live recordings from user device).

##### Watch out for this space!

### Project Files
- [CRNN_model.py]() - Convolutional recurrent neural network (CRNN) model

- [CRNN_validation_test.py] - Validates saved trained CRNN model against test data and displays accuracy results

- [RF_KNN_enssemble_model.py] - Random Forest and KNN Ensemble model (baseline)

- [get_swing_prediction.py] - Predicts swing dance style for a single song with playlist option for users

- [playlist_generator.py] - Generates a playlist based on a dance style and given playlist duration

- [swing_band_name_generator.py] - Just for Fun! Generates Random jazz-era band name based on artist names in the dataset

### References
