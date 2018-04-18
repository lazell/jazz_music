

![Lindy Hop](img/header.png?raw=true "LindyList")
## Lindy List - Swing Dance Music Classifier

LindyList is a music classifier that analyzes a jazz-era
song to determine the most suitable style(s) of swing dance
from the five predominant styles:

* Lindy Hop
* Slow Swing/Drag Blues
* Balboa
* Charleston
* St Louis and Collegiate Shag


### Introduction
Auto-tagging music genre has been an active area of deep learning research in recent years. The LindyList, uses many of those same approaches, but instead classifies by the most suitable style of swing dance from the five most common styles danced today. The prediction(s) can be used as a feature to improve song discovery and quality of playlist and song recommendations.

The LindyList is for jazz-era music buffs, jazz musicians, DJs, performers, social swing dancers and film/TV sound design. Dance-style knowledge would typically be acquired from years of exposure to a variety of jazz music and dance styles. With this project, I intend to make swing music discovery more accessible and engaging from the dancer’s perspective.

### The Dataset
The data set consists of 3000 unique songs, recorded between 1926-1959. Each was randomly selected
and downloaded from [Jazz On Line](https://http://www.jazz-on-line.com/), a public domain jazz
music website. Since labels for dance styles did not exist for
this dataset, I manually labelled songs by dance style and annotating each song with notes on whether a song was swing danceable, whether it
was particularly infectious, and whether the quality of the
recording was good. Since I labelled the dataset myself,
the dataset is undoubtedly biased toward my own
in swing dancing interpretations. Out of the 3000
songs, 1376 were “swing-danceable,” and were used for
training and validation testing.

### The Approach
This project has two components:  the Ensemble
model and the Neural Net model. To build a baseline to
model, I first took the Ensemble
model approach. I extracted from each song harmonic and;
percussive tempo, total beat count per song, 12 pitch prominence
scores, and relative root mean square energy values. I then
prototyped various machine learning models and compared
for accuracy:

* Random Forest Classifier
* Gradient Boosted Classifier
* K Nearest Neighbors

From there, I generated multiple 30-second samples of each song and generated the log-power mel-spectrogram arrays to use as training inputs for the artificial neural network models:

* Convolutional Neural Nets
* Convolutional Recurrent Neural Nets

![Lindy Hop](img/mel-specs.png?raw=true "Lindy Hop 1")

In doing so, I discovered that certain classes were difficult to
predict and eventually decided to remove the slow
swing/drag blues category from the neural network models since it had
too much variability in the sub-category.


## Results

### Ensemble Model Results
The ensemble model, which consisted of two differently optimized random forest classifiers and one k nearest neighbors classifier achieved 50% accuracy across 5 classes and 60.9% accuracy across 4 classes (when excluding slow swing /drag blues)

![Ensemble Results](img/Ensemble_results.png?raw=true "Results Model 1")

### Neural Net Model Results
Using a Convolutional Recurrent Neural Network I was able to achieve 81% using a 4-layer deep convolutional recurrent neural network across the same 4 classes.

More importantly, the recall results for individual dance styles had increased to over 52%, with Charleston and Shag styles performing particularly well compared to the ensemble baseline.

By using a neural network model, I was able to reduce prediction time to approximately the tenth of the time it takes the ensemble model to predict (both times are inclusive of audio pre-processing).

![CRNN Results](img/CRNN_results.png?raw=true "Results Model 2")


## Real World Testing
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
- [CRNN_model.py](https://github.com/lazell/jazz_music/blob/master/CRNN_model.py) - Convolutional recurrent neural network (CRNN) model.

- [CRNN_validation_test.py](https://github.com/lazell/jazz_music/blob/master/CRNN_validation_test.py) - Validates saved trained CRNN model against test data and displays accuracy results.

- [RF_KNN_ensemble_model.py](https://github.com/lazell/jazz_music/blob/master/RF_KNN_ensemble_model.py)  - Random Forest and KNN Ensemble model (baseline).

- [get_swing_prediction.py](https://github.com/lazell/jazz_music/blob/master/get_swing_prediction.py)  - Predicts swing dance style for a single song file (mp3) with the option to recommend a dance playlist.

- [playlist_generator.py](https://github.com/lazell/jazz_music/blob/master/playlist_generator.py)  - Generates a 30 minute playlist based on a dance style with songs in the dataset.

- [swing_band_generator.py](https://github.com/lazell/jazz_music/blob/master/swing_band_generator.py)  - Just for Fun! Randomlt generates jazz-era style band names based on artist in the dataset.

![Lindy Hop](img/jean_lindy.gif?raw=true "Lindy Hop 1")

### References

1. [https://pdfs.semanticscholar.org](https://pdfs.semanticscholar.org/b9ba/8c4a00f5ee43e768db2acc8b56f017176f3e.pdf) AUTOMATIC TAGGING USING DEEP CONVOLUTIONAL NEURAL NETWORKS - Keunwoo Choi, Gyorgy Fazekas, Mark Sandler

2. [http://www.iaeng.org](http://www.iaeng.org/publication/IMECS2010/IMECS2010_pp546-550.pdf) Automatic Musical Pattern Feature Extraction Using Convolutional Neural Network - Tom LH. Li, Antoni B. Chan and Andy HW. Chun

3. [http://music.ece.drexel.edu](http://music.ece.drexel.edu/research/MusicalAttributes/ModelingGenre) Modeling Genre with Musical Attributes - MetLab

4. [https://chatbotslife.com](https://chatbotslife.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194) Finding The Genre of a Song Using Deep Learning - AI Odyssey

5. [http://image-net.org/](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf) ELU-Networks:
 Fast and Accurate CNN Learning on ImageNet - Johannes Kelper University Linz

6. [https://github.com/drscotthawley/audio-classifier-keras-cnn](https://github.com/drscotthawley/audio-classifier-keras-cnn) Audio Classifier Keras using Convolutional Neural Networks - Scott Hawley

7. [https://github.com/meetshah1995/crnn-music-genre-classification](https://github.com/meetshah1995/crnn-music-genre-classification) CRNN Music Genre Classification - Pragnesh Shah
